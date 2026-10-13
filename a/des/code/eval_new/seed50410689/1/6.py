from __future__ import annotations

import random

import numpy as np


BG = 0
OCCLUDER = 2
PATTERN_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
GENERATION_RETRIES = 500


def _visible_pattern_color(grid: np.ndarray) -> int:
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, OCCLUDER)]
    assert len(colors) == 1, "expected exactly one visible bottom-layer color"
    return colors[0]


def _render_template(template: np.ndarray, height: int, width: int, color: int) -> np.ndarray:
    pr, pc = map(int, template.shape)
    rows = np.arange(height) % pr
    cols = np.arange(width) % pc
    mask = template[rows[:, None], cols[None, :]]
    out = np.zeros((height, width), dtype=int)
    out[mask] = int(color)
    return out


def _fit_candidate(observed: np.ndarray, observed_values: np.ndarray, pr: int, pc: int) -> np.ndarray | None:
    if pr * pc > int(observed.sum()):
        return None

    template = np.full((pr, pc), -1, dtype=np.int8)
    seen = np.zeros((pr, pc), dtype=bool)
    rows, cols = np.nonzero(observed)
    for r, c in zip(rows.tolist(), cols.tolist()):
        rr = r % pr
        cc = c % pc
        value = int(observed_values[r, c])
        current = int(template[rr, cc])
        if current == -1:
            template[rr, cc] = value
            seen[rr, cc] = True
        elif current != value:
            return None

    # The puzzle is only accepted when the visible windows jointly cover every residue class.
    if not bool(seen.all()):
        return None
    return template.astype(bool)


def _solve_periodic_pattern(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    color = _visible_pattern_color(g)

    observed = g != OCCLUDER
    observed_values = (g == color).astype(np.int8)

    best_key: tuple[int, int, int] | None = None
    best_out: np.ndarray | None = None
    visible = int(observed.sum())

    for pr in range(1, height + 1):
        for pc in range(1, width + 1):
            if pr * pc > visible:
                break
            template = _fit_candidate(observed, observed_values, pr, pc)
            if template is None:
                continue
            candidate = _render_template(template, height, width, color)
            if not np.array_equal(candidate[observed], g[observed]):
                continue
            key = (pr * pc, pr, pc)
            if best_key is None or key < best_key:
                best_key = key
                best_out = candidate

    assert best_out is not None, "failed to infer a fully observed periodic template"
    return best_out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the red occluder and reconstruct the full periodic pattern hidden underneath.

    Input structure:
    - Background color is black.
    - One other color forms a repeating bottom-layer pattern across the whole grid.
    - A red occluder covers most of the grid, but small rectangular windows and some exposed edge areas reveal fragments of the bottom layer.

    Rule:
    - Treat every non-red cell as a visible sample of the hidden binary pattern.
    - Infer the smallest row/column period and phase whose repeated template matches all visible samples.
    - Remove every red cell and fill the whole grid with the inferred periodic pattern, keeping the already visible fragments unchanged.

    Key insight:
    - The windows are sparse observations of one globally periodic template; once all period residue classes are identified, the full hidden layer can be tiled back over the entire grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve_periodic_pattern(g)


def _has_period(template: np.ndarray, pr: int, pc: int) -> bool:
    height, width = map(int, template.shape)
    for r in range(height):
        for c in range(width):
            if bool(template[r, c]) != bool(template[r % pr, c % pc]):
                return False
    return True


def _is_exact_template(template: np.ndarray) -> bool:
    height, width = map(int, template.shape)
    if bool(template.all()) or not bool(template.any()):
        return False
    for pr in range(1, height + 1):
        for pc in range(1, width + 1):
            if pr == height and pc == width:
                continue
            if _has_period(template, pr, pc):
                return False
    return True


def _run_sequence(length: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, length), rng.randint(1, min(3, length - 1))))
        starts = [0] + cuts
        ends = cuts + [length]
        current = rng.choice([False, True])
        values: list[bool] = []
        for a, b in zip(starts, ends):
            values.extend([current] * (b - a))
            current = not current
        seq = np.asarray(values, dtype=bool)
        if _is_exact_template(seq[None, :]):
            return seq
    raise RuntimeError("failed to build exact-period stripe sequence")


def _sample_template(rng: random.Random) -> np.ndarray:
    families = ("vertical", "horizontal", "checker", "dots", "block")
    family = rng.choice(families)

    if family == "vertical":
        width = rng.randint(2, 5)
        return _run_sequence(width, rng)[None, :]

    if family == "horizontal":
        height = rng.randint(2, 5)
        return _run_sequence(height, rng)[:, None]

    if family == "checker":
        rows, cols = np.indices((2, 2))
        return ((rows + cols) % 2 == 0)

    if family == "dots":
        for _ in range(200):
            pr = rng.randint(2, 4)
            pc = rng.randint(2, 4)
            template = np.zeros((pr, pc), dtype=bool)
            n_dots = 1 if rng.random() < 0.7 else 2
            positions = rng.sample([(r, c) for r in range(pr) for c in range(pc)], n_dots)
            for r, c in positions:
                template[r, c] = True
            if _is_exact_template(template):
                return template
        raise RuntimeError("failed to build dot template")

    for _ in range(200):
        pr = rng.randint(2, 5)
        pc = rng.randint(2, 5)
        bh = rng.randint(1, pr - 1)
        bw = rng.randint(1, pc - 1)
        r0 = rng.randint(0, pr - bh)
        c0 = rng.randint(0, pc - bw)
        template = np.zeros((pr, pc), dtype=bool)
        template[r0 : r0 + bh, c0 : c0 + bw] = True
        if _is_exact_template(template):
            return template
    raise RuntimeError("failed to build block template")


def _tile_with_phase(template: np.ndarray, height: int, width: int, color: int, phase_r: int, phase_c: int) -> np.ndarray:
    pr, pc = map(int, template.shape)
    rows = (np.arange(height) + phase_r) % pr
    cols = (np.arange(width) + phase_c) % pc
    mask = template[rows[:, None], cols[None, :]]
    out = np.zeros((height, width), dtype=int)
    out[mask] = int(color)
    return out


def _halo_cells(mask: np.ndarray) -> np.ndarray:
    height, width = map(int, mask.shape)
    halo = np.zeros_like(mask, dtype=bool)
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if bool(mask[r - 1 : r + 2, c - 1 : c + 2].all()):
                halo[r, c] = True
    return halo


def _make_occluder_mask(height: int, width: int, need_cells: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(120):
        mask = np.zeros((height, width), dtype=bool)

        main_h = rng.randint(max(6, height // 2), height - 2)
        main_w = rng.randint(max(6, width // 2), width - 2)
        r0 = rng.randint(1, height - main_h - 1)
        c0 = rng.randint(1, width - main_w - 1)
        r1 = r0 + main_h
        c1 = c0 + main_w
        mask[r0:r1, c0:c1] = True

        for _ in range(rng.randint(1, 3)):
            side = rng.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                aw = rng.randint(max(3, main_w // 3), main_w)
                ah = rng.randint(2, max(2, height // 4))
                ac0 = rng.randint(c0, c1 - aw)
                if side == "top":
                    ar0 = max(0, r0 - ah)
                    ar1 = r0 + rng.randint(1, min(3, main_h - 1))
                else:
                    ar0 = r1 - rng.randint(1, min(3, main_h - 1))
                    ar1 = min(height, r1 + ah)
                mask[ar0:ar1, ac0 : ac0 + aw] = True
            else:
                ah = rng.randint(max(3, main_h // 3), main_h)
                aw = rng.randint(2, max(2, width // 4))
                ar0 = rng.randint(r0, r1 - ah)
                if side == "left":
                    ac0 = max(0, c0 - aw)
                    ac1 = c0 + rng.randint(1, min(3, main_w - 1))
                else:
                    ac0 = c1 - rng.randint(1, min(3, main_w - 1))
                    ac1 = min(width, c1 + aw)
                mask[ar0 : ar0 + ah, ac0:ac1] = True

        for _ in range(rng.randint(0, 2)):
            rows, cols = np.nonzero(mask)
            if rows.size == 0:
                break
            rb0, rb1 = int(rows.min()), int(rows.max())
            cb0, cb1 = int(cols.min()), int(cols.max())
            if rb1 - rb0 < 4 or cb1 - cb0 < 4:
                break
            side = rng.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                bite_h = rng.randint(1, max(1, (rb1 - rb0 + 1) // 4))
                bite_w = rng.randint(2, max(2, (cb1 - cb0 + 1) // 3))
                bc0 = rng.randint(cb0, cb1 - bite_w + 1)
                br0 = rb0 if side == "top" else rb1 - bite_h + 1
                mask[br0 : br0 + bite_h, bc0 : bc0 + bite_w] = False
            else:
                bite_h = rng.randint(2, max(2, (rb1 - rb0 + 1) // 3))
                bite_w = rng.randint(1, max(1, (cb1 - cb0 + 1) // 4))
                br0 = rng.randint(rb0, rb1 - bite_h + 1)
                bc0 = cb0 if side == "left" else cb1 - bite_w + 1
                mask[br0 : br0 + bite_h, bc0 : bc0 + bite_w] = False

        coverage = float(mask.mean())
        if not (0.48 <= coverage <= 0.82):
            continue
        if int(_halo_cells(mask).sum()) < int(need_cells):
            continue
        return mask
    return None


def _window_rect_ok(mask: np.ndarray, r0: int, c0: int, hh: int, ww: int) -> bool:
    height, width = map(int, mask.shape)
    if r0 <= 0 or c0 <= 0 or r0 + hh >= height or c0 + ww >= width:
        return False
    return bool(mask[r0 - 1 : r0 + hh + 1, c0 - 1 : c0 + ww + 1].all())


def _carve_windows(mask: np.ndarray, pr: int, pc: int, rng: random.Random) -> np.ndarray | None:
    working = mask.copy()
    halo = _halo_cells(mask)
    coords = list(zip(*np.nonzero(halo)))
    rng.shuffle(coords)

    by_residue: dict[tuple[int, int], list[tuple[int, int]]] = {
        (rr, cc): [] for rr in range(pr) for cc in range(pc)
    }
    for r, c in coords:
        by_residue[(int(r % pr), int(c % pc))].append((int(r), int(c)))

    for rr in range(pr):
        for cc in range(pc):
            candidates = by_residue[(rr, cc)]
            if not candidates:
                return None
            r, c = candidates[0]
            working[r, c] = False

    for _ in range(rng.randint(2, 5)):
        carved = False
        for hh, ww in rng.sample([(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)], 6):
            for _ in range(20):
                r0 = rng.randint(1, working.shape[0] - hh - 1)
                c0 = rng.randint(1, working.shape[1] - ww - 1)
                if _window_rect_ok(working, r0, c0, hh, ww):
                    working[r0 : r0 + hh, c0 : c0 + ww] = False
                    carved = True
                    break
            if carved:
                break
    return working


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(12, 24)
    width = rng.randint(12, 24)

    template = _sample_template(rng)
    pr, pc = map(int, template.shape)
    color = rng.choice(PATTERN_COLORS)
    phase_r = rng.randrange(pr)
    phase_c = rng.randrange(pc)
    output = _tile_with_phase(template, height, width, color, phase_r, phase_c)

    need_cells = pr * pc + 4
    mask = _make_occluder_mask(height, width, need_cells, rng)
    if mask is None:
        return None
    mask = _carve_windows(mask, pr, pc, rng)
    if mask is None:
        return None

    grid = output.copy()
    grid[mask] = OCCLUDER

    if not np.any(grid == color):
        return None
    if np.array_equal(grid, output):
        return None
    if np.array_equal(main(grid), output):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-occlusion puzzle")
