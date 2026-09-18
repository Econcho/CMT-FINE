"""Synthetic NumPy checks for the CMT research specification, not a detector.

No dataset access, training, network connection, or learned performance claims.
Run with: python verify_cmt_math.py --output cmt_math_verification.json
"""
import argparse
import json
from pathlib import Path
import numpy as np


def points(height, width):
    yy, xx = np.mgrid[:height, :width]
    return np.stack([xx + 0.5, yy + 0.5], axis=-1)


def pack(evidence, xy, origin):
    d = xy - np.asarray(origin)
    mass = evidence.sum()
    p = (evidence[..., None] * d).sum(axis=(0, 1))
    q = np.einsum('hw,hwi,hwj->ij', evidence, d, d)
    return np.array([mass, p[0], p[1], q[0, 0], q[0, 1], q[1, 1]])


def unpack(state):
    return state[0], state[1:3], np.array([[state[3], state[4]], [state[4], state[5]]])


def move(state, displacement):
    m, p, q = unpack(state)
    d = np.asarray(displacement)
    out_p = p + d * m
    out_q = q + np.outer(p, d) + np.outer(d, p) + m * np.outer(d, d)
    return np.array([m, *out_p, out_q[0, 0], out_q[0, 1], out_q[1, 1]])


def encode(evidence, stride):
    h, w = evidence.shape
    assert h % stride == w % stride == 0
    xy = points(h, w)
    state = np.zeros((h // stride, w // stride, 6), dtype=np.float64)
    for i in range(h // stride):
        for j in range(w // stride):
            sl = (slice(i * stride, (i + 1) * stride), slice(j * stride, (j + 1) * stride))
            state[i, j] = pack(evidence[sl], xy[sl], ((j + .5) * stride, (i + .5) * stride))
    return state


def merge(states, child_stride):
    h, w, _ = states.shape
    assert h % 2 == w % 2 == 0
    out = np.zeros((h // 2, w // 2, 6), dtype=np.float64)
    for i in range(h // 2):
        for j in range(w // 2):
            parent_o = np.array([(j + .5) * 2 * child_stride, (i + .5) * 2 * child_stride])
            for di in range(2):
                for dj in range(2):
                    ci, cj = 2 * i + di, 2 * j + dj
                    child_o = np.array([(cj + .5) * child_stride, (ci + .5) * child_stride])
                    out[i, j] += move(states[ci, cj], child_o - parent_o)
    return out


def kernel(xy, r, a):
    z = (xy - r) / a
    return np.prod(np.maximum(1 - z * z, 0) ** 2, axis=-1)


def centroid(state, origin):
    return np.asarray(origin) + state[1:3] / state[0]


def read(evidence, stride, r, a):
    states = encode(evidence, stride)
    h, w = evidence.shape
    xy = points(h, w)
    origins = points(h // stride, w // stride) * stride
    weights = kernel(origins, r, a)
    step_weights = weights.repeat(stride, 0).repeat(stride, 1)
    aggregate = np.zeros(6)
    eps_d, eps_n = 0., 0.
    lipschitz = 8 / (3 * np.sqrt(3)) * np.linalg.norm(1 / a)
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            state = states[i, j]
            d = origins[i, j] - r
            aggregate += weights[i, j] * move(state, d)
            # Including all cells is conservative and also includes support boundary cells.
            trace = state[3] + state[5]
            spread = np.sqrt(max(state[0] * trace, 0.))
            eps_d += lipschitz * spread
            eps_n += lipschitz * (np.linalg.norm(d) * spread + trace)
    direct_step = pack(evidence * step_weights, xy, r)
    direct_smooth = pack(evidence * kernel(xy, r, a), xy, r)
    err = np.linalg.norm(centroid(aggregate, r) - centroid(direct_smooth, r))
    informative = aggregate[0] > eps_d
    bound = ((eps_n + np.linalg.norm(aggregate[1:3] / aggregate[0]) * eps_d)
             / (aggregate[0] - eps_d)) if informative else None
    return aggregate, direct_step, direct_smooth, eps_d, eps_n, err, bound


def encoding_matrix(stride):
    d = (points(stride, stride) - stride / 2).reshape(-1, 2)
    x, y = d.T
    return np.stack([np.ones(stride ** 2), x, y, x*x, x*y, y*y])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('cmt_math_verification.json'))
    args = parser.parse_args()
    rng = np.random.default_rng(20260913)
    results = {'seed': 20260913, 'dtype': 'float64', 'numpy': np.__version__,
               'scope': 'synthetic algebra only; no detector, training, or real-data evaluation'}
    x = rng.random((32, 48))
    merged = merge(merge(merge(encode(x, 1), 1), 2), 4)
    direct = encode(x, 8)
    results['hierarchical_merge_max_abs_error'] = float(np.max(np.abs(merged - direct)))
    assert np.allclose(merged, direct, atol=1e-9, rtol=1e-12)

    state = pack(x, points(*x.shape), [6.3, -4.2])
    via = move(move(state, [1.3, -2.1]), [-4.7, 9.1])
    once = move(state, [-3.4, 7.0])
    results['origin_composition_max_abs_error'] = float(np.max(np.abs(via-once)))
    assert np.allclose(via, once, atol=1e-8, rtol=1e-12)

    small = np.zeros((64, 64))
    small[20:28, 21:29] = rng.random((8, 8))
    initial = pack(small, points(64, 64), [0, 0])
    center_errors, cov_errors, masses = [], [], []
    m, p, q = unpack(initial)
    cov = q / m - np.outer(p/m, p/m)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            shifted = np.roll(small, (dy, dx), axis=(0, 1))
            # Support is at least 17 pixels from border; roll wraps only zeros.
            got = pack(shifted, points(64, 64), [0, 0])
            center_errors.append(np.linalg.norm(centroid(got, [0, 0]) - centroid(initial, [0, 0]) - [dx, dy]))
            mg, pg, qg = unpack(got)
            cov_errors.append(np.max(np.abs(qg/mg - np.outer(pg/mg, pg/mg) - cov)))
            masses.append(abs(mg - m))
    results['full_support_translation_49_shifts'] = {
        'max_centroid_error_px': float(max(center_errors)),
        'max_covariance_error_px2': float(max(cov_errors)),
        'max_mass_error': float(max(masses))}
    assert max(center_errors) < 1e-10 and max(cov_errors) < 1e-9

    max_step_err = max_d_ratio = max_n_ratio = 0.
    bound_checks, bound_rows = 0, []
    for case in range(30):
        r = np.array([24., 24.]) + rng.normal(0, 2, 2)
        a = rng.uniform(22, 40, size=2)
        got, step, exact, ed, en, error, bound = read(small, 4, r, a)
        max_step_err = max(max_step_err, np.max(np.abs(got-step)))
        max_d_ratio = max(max_d_ratio, abs(got[0]-exact[0])/ed)
        max_n_ratio = max(max_n_ratio, np.linalg.norm(got[1:3]-exact[1:3])/en)
        assert np.allclose(got, step, atol=1e-9)
        assert abs(got[0]-exact[0]) <= ed + 1e-10
        assert np.linalg.norm(got[1:3]-exact[1:3]) <= en + 1e-10
        if bound is not None:
            bound_checks += 1
            assert error <= bound + 1e-10
            bound_rows.append({'centroid_error_px': float(error), 'upper_bound_px': float(bound)})
    results['piecewise_constant_readout_max_abs_error'] = float(max_step_err)
    results['smooth_kernel_bound_checks'] = {
        'cases': 30, 'informative_centroid_bounds': bound_checks,
        'max_mass_error_over_bound': float(max_d_ratio),
        'max_first_moment_error_over_bound': float(max_n_ratio),
        'cases_with_centroid_bound': bound_rows}
    assert bound_checks == 30

    ranks = {str(s): int(np.linalg.matrix_rank(encoding_matrix(s))) for s in (1, 2, 4, 8)}
    results['six_moment_encoding_ranks'] = ranks
    assert ranks == {'1': 1, '2': 4, '4': 6, '8': 6}
    e2 = rng.random((2, 2))
    m2 = pack(e2, points(2, 2), [1, 1])
    signs = 2 * (points(2, 2) - 1)
    reconstructed = m2[0]/4 + signs[..., 0]*m2[1]/2 + signs[..., 1]*m2[2]/2 + signs[..., 0]*signs[..., 1]*m2[4]
    results['stride2_inverse_max_abs_error'] = float(np.max(np.abs(e2-reconstructed)))
    assert np.allclose(e2, reconstructed, atol=1e-12)

    k4 = encoding_matrix(4)
    _, _, vh = np.linalg.svd(k4, full_matrices=True)
    null = vh[-1]
    evidence_a = np.ones(16)
    evidence_b = evidence_a + .5 * null / np.max(np.abs(null))
    assert np.all(evidence_b > 0)
    same_error = np.max(np.abs(k4 @ evidence_a - k4 @ evidence_b))
    results['stride4_information_loss_counterexample'] = {
        'different_positive_evidence_l1': float(np.abs(evidence_a-evidence_b).sum()),
        'six_moment_max_abs_difference': float(same_error)}
    assert same_error < 1e-10

    target_mass, bg_mass = 3., 1.
    target_center, bg_center = np.array([4., 5.]), np.array([14., 11.])
    mix_center = (target_mass*target_center+bg_mass*bg_center)/(target_mass+bg_mass)
    bias = bg_mass/(target_mass+bg_mass)*(bg_center-target_center)
    assert np.allclose(mix_center-target_center, bias)
    results['background_bias_identity_max_error'] = float(np.max(np.abs(mix_center-target_center-bias)))
    results['status'] = 'PASS'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in results.items() if k != 'smooth_kernel_bound_checks'}, indent=2))
    print('smooth_kernel_bound_checks:', bound_checks, '/ 30')


if __name__ == '__main__':
    main()
