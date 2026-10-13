from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7
FILTER_COLORS = (1, 2, 3, 6, 8, 9)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = DIR4 if connectivity == 4 else DIR8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _reflect(kind: str, dr: int, dc: int) -> tuple[int, int]:
    if kind == "\\":
        return dc, dr
    return -dc, -dr


def _mirror_kinds(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirrors: dict[tuple[int, int], str] = {}
    for comp in _components(grid == GREY, connectivity=8):
        diffs = {r - c for r, c in comp}
        sums = {r + c for r, c in comp}
        if len(diffs) == 1:
            kind = "\\"
        elif len(sums) == 1:
            kind = "/"
        else:
            raise AssertionError("grey component is not a diagonal mirror")
        for cell in comp:
            mirrors[cell] = kind
    return mirrors


def _targets(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    groups = _components(grid == ORANGE, connectivity=4)
    by_cell: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(groups):
        for cell in comp:
            by_cell[cell] = idx
    return groups, by_cell


def _source_direction(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    sides = int(on_top) + int(on_bottom) + int(on_left) + int(on_right)
    assert sides == 1, "sources must be on a unique edge"
    if on_top:
        return 1, 0
    if on_bottom:
        return -1, 0
    if on_left:
        return 0, 1
    return 0, -1


def _iter_sources(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    sources: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != YELLOW:
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            dr, dc = _source_direction(r, c, h, w)
            sources.append((r, c, dr, dc))
    sources.sort(key=lambda item: (item[1], item[0]))
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    mirror_at = _mirror_kinds(g)
    target_groups, target_of = _targets(g)
    out = g.copy()

    for sr, sc, dr, dc in _iter_sources(g):
        r, c = sr, sc
        last_filter: int | None = None
        seen: set[tuple[int, int, int, int]] = {(r, c, dr, dc)}

        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break

            state = (r, c, dr, dc)
            if state in seen:
                break
            seen.add(state)

            value = int(g[r, c])
            if value == BLACK:
                out[r, c] = YELLOW
                continue
            if value == GREY:
                dr, dc = _reflect(mirror_at[(r, c)], dr, dc)
                continue
            if value == ORANGE:
                if last_filter is not None:
                    for tr, tc in target_groups[target_of[(r, c)]]:
                        out[tr, tc] = last_filter
                break
            if value != YELLOW:
                last_filter = value

    return out


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    cells: list[tuple[int, int]] = []
    if sr == er:
        step = 1 if ec > sc else -1
        for c in range(sc + step, ec + step, step):
            cells.append((sr, c))
        return cells
    if sc == ec:
        step = 1 if er > sr else -1
        for r in range(sr + step, er + step, step):
            cells.append((r, sc))
        return cells
    raise AssertionError("segments must be axis-aligned")


def _choice_range(lo: int, hi: int) -> list[int]:
    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _random_target_rows(hit_row: int, height: int, grid_h: int) -> list[int]:
    lo = max(1, hit_row - height + 1)
    hi = min(hit_row, grid_h - height - 2)
    return _choice_range(lo, hi)


def _random_target_cols(hit_col: int, width: int, strip_l: int, strip_r: int) -> list[int]:
    lo = max(strip_l, hit_col - width + 1)
    hi = min(hit_col, strip_r - width + 1)
    return _choice_range(lo, hi)


def _target_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def _pick_mirror_footprint(
    kind: str,
    pivot: tuple[int, int],
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    pr, pc = pivot
    if kind == "\\":
        patterns = [
            {(pr, pc), (pr + 1, pc + 1)},
            {(pr - 1, pc - 1), (pr, pc)},
            {(pr - 1, pc - 1), (pr, pc), (pr + 1, pc + 1)},
        ]
    else:
        patterns = [
            {(pr, pc), (pr + 1, pc - 1)},
            {(pr - 1, pc + 1), (pr, pc)},
            {(pr - 1, pc + 1), (pr, pc), (pr + 1, pc - 1)},
        ]
    valid = []
    for cells in patterns:
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells) and not (cells & blocked):
            valid.append(cells)
    if not valid:
        return None
    return set(random.choice(valid))


def _internal_cells(segments: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for seg in segments:
        if len(seg) >= 3:
            out.extend(seg[1:-1])
    return out


def _nbr4(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = cell
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _select_filter_cells(
    ordered_cells: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    special_cells: set[tuple[int, int]],
    wanted: int,
) -> list[tuple[int, int]]:
    ranked = sorted(candidates, key=ordered_cells.index)
    picked: list[tuple[int, int]] = []
    blocked = set(special_cells)
    for cell in ranked:
        if cell in blocked:
            continue
        if any(nei in blocked for nei in _nbr4(cell)):
            continue
        picked.append(cell)
        blocked.add(cell)
        blocked.update(_nbr4(cell))
        if len(picked) == wanted:
            break
    return picked


def _build_ray(strip_l: int, strip_r: int, h: int, template: str) -> dict | None:
    source_col_options = _choice_range(strip_l + 2, strip_r - 2)
    if not source_col_options:
        return None
    src_c = random.choice(source_col_options)
    source = (0, src_c)

    if template == "straight_hit":
        tw = random.randint(2, 3)
        th = random.randint(2, 3)
        top_options = _choice_range(4, h - th - 2)
        col_options = _random_target_cols(src_c, tw, strip_l, strip_r)
        if not top_options or not col_options:
            return None
        top = random.choice(top_options)
        left = random.choice(col_options)
        target = _target_cells(top, left, th, tw)
        segments = [_line_cells(source, (top, src_c))]
        return {
            "source": source,
            "segments": segments,
            "mirrors": [],
            "target": target,
            "hit": True,
        }

    if template == "straight_miss":
        segments = [_line_cells(source, (h - 1, src_c))]
        return {
            "source": source,
            "segments": segments,
            "mirrors": [],
            "target": None,
            "hit": False,
        }

    turn_dir = random.choice([-1, 1])
    mirror_kind = "/" if turn_dir < 0 else "\\"

    if template == "one_turn_hit":
        tw = random.randint(2, 3)
        th = random.randint(2, 3)
        row_options = _choice_range(3, h - 4)
        if not row_options:
            return None
        random.shuffle(row_options)
        for row in row_options:
            if turn_dir > 0:
                col_options = _choice_range(src_c + 2, strip_r - tw + 1)
            else:
                col_options = _choice_range(strip_l, src_c - tw - 1)
            top_options = _random_target_rows(row, th, h)
            if not col_options or not top_options:
                continue
            left = random.choice(col_options)
            top = random.choice(top_options)
            entry = (row, left) if turn_dir > 0 else (row, left + tw - 1)
            target = _target_cells(top, left, th, tw)
            segments = [_line_cells(source, (row, src_c)), _line_cells((row, src_c), entry)]
            mirrors = [{"pivot": (row, src_c), "kind": mirror_kind}]
            return {
                "source": source,
                "segments": segments,
                "mirrors": mirrors,
                "target": target,
                "hit": True,
            }
        return None

    row_hi = h - 7 if template == "two_turn_hit" else h - 5
    row_options = _choice_range(3, row_hi)
    if not row_options:
        return None
    random.shuffle(row_options)
    for row in row_options:
        if turn_dir > 0:
            mid_cols = _choice_range(src_c + 3, strip_r - 1)
        else:
            mid_cols = _choice_range(strip_l + 1, src_c - 3)
        if not mid_cols:
            continue
        mid_c = random.choice(mid_cols)
        segments = [_line_cells(source, (row, src_c)), _line_cells((row, src_c), (row, mid_c))]
        mirrors = [
            {"pivot": (row, src_c), "kind": mirror_kind},
            {"pivot": (row, mid_c), "kind": mirror_kind},
        ]
        if template == "two_turn_miss":
            segments.append(_line_cells((row, mid_c), (h - 1, mid_c)))
            return {
                "source": source,
                "segments": segments,
                "mirrors": mirrors,
                "target": None,
                "hit": False,
            }

        tw = random.randint(2, 3)
        th = random.randint(2, 3)
        top_options = _choice_range(row + 2, h - th - 2)
        col_options = _random_target_cols(mid_c, tw, strip_l, strip_r)
        if not top_options or not col_options:
            continue
        top = random.choice(top_options)
        left = random.choice(col_options)
        target = _target_cells(top, left, th, tw)
        segments.append(_line_cells((row, mid_c), (top, mid_c)))
        return {
            "source": source,
            "segments": segments,
            "mirrors": mirrors,
            "target": target,
            "hit": True,
        }
    return None


def _free_rect(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    top: int,
    left: int,
    height: int,
    width: int,
) -> bool:
    h, w = grid.shape
    if top <= 0 or left <= 0 or top + height >= h or left + width >= w:
        return False
    cells = _target_cells(top, left, height, width)
    return not (cells & reserved) and not (cells & occupied)


def _rotate_scene(grid: np.ndarray) -> np.ndarray:
    k = random.randint(0, 3)
    out = np.rot90(grid, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    n_rays = random.randint(1, 3)
    if n_rays == 1:
        templates = [random.choice(["one_turn_hit", "two_turn_hit", "two_turn_miss"])]
    else:
        pool = ["straight_hit", "straight_miss", "one_turn_hit", "two_turn_hit", "two_turn_miss"]
        templates = [random.choice(pool) for _ in range(n_rays)]
        if not any(t.endswith("hit") for t in templates):
            templates[random.randrange(n_rays)] = random.choice(["straight_hit", "one_turn_hit", "two_turn_hit"])
        if not any("turn" in t for t in templates):
            templates[random.randrange(n_rays)] = random.choice(["one_turn_hit", "two_turn_hit", "two_turn_miss"])

    strip_widths = [random.randint(7, 10) for _ in range(n_rays)]
    gaps = [random.randint(1, 2) for _ in range(n_rays - 1)]
    margins = (random.randint(1, 2), random.randint(1, 2))
    width = margins[0] + sum(strip_widths) + sum(gaps) + margins[1]
    height = random.randint(15, 23)
    if width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    reserved_path: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    hit_targets = 0

    strips: list[tuple[int, int]] = []
    col = margins[0]
    for idx, sw in enumerate(strip_widths):
        strips.append((col, col + sw - 1))
        col += sw
        if idx < len(gaps):
            col += gaps[idx]

    palette = random.sample(FILTER_COLORS, k=random.randint(3, len(FILTER_COLORS)))

    for (strip_l, strip_r), template in zip(strips, templates):
        spec = _build_ray(strip_l, strip_r, height, template)
        if spec is None:
            return None

        target = spec["target"]
        target_cells = set() if target is None else set(target)
        ordered_cells = [cell for seg in spec["segments"] for cell in seg]
        if reserved_path & set(ordered_cells):
            return None

        mirror_footprints: list[set[tuple[int, int]]] = []
        blocked = occupied | target_cells | {spec["source"]}
        for mirror in spec["mirrors"]:
            footprint = _pick_mirror_footprint(mirror["kind"], mirror["pivot"], height, width, blocked)
            if footprint is None:
                return None
            mirror_footprints.append(footprint)
            blocked |= footprint

        filter_candidates = [cell for cell in _internal_cells(spec["segments"]) if cell not in target_cells]
        random.shuffle(filter_candidates)
        if spec["hit"]:
            n_filters = random.randint(0, min(3, len(filter_candidates)))
        else:
            n_filters = random.randint(0, min(2, len(filter_candidates)))
        special_cells = set().union({spec["source"]}, target_cells, *mirror_footprints)
        chosen_filters = _select_filter_cells(ordered_cells, filter_candidates, special_cells, n_filters)

        grid[spec["source"]] = YELLOW
        occupied.add(spec["source"])

        for footprint in mirror_footprints:
            for r, c in footprint:
                grid[r, c] = GREY
            occupied |= footprint

        if target is not None:
            for r, c in target_cells:
                grid[r, c] = ORANGE
            occupied |= target_cells
            hit_targets += 1

        for cell in chosen_filters:
            color = random.choice(palette)
            grid[cell] = color
            occupied.add(cell)

        reserved_path |= set(ordered_cells)

    if hit_targets == 0:
        return None

    decoy_target_count = random.randint(0, 2)
    for _ in range(decoy_target_count):
        placed = False
        for _retry in range(50):
            th = random.randint(2, 3)
            tw = random.randint(2, 3)
            top = random.randint(1, height - th - 2)
            left = random.randint(1, width - tw - 2)
            if not _free_rect(grid, reserved_path, occupied, top, left, th, tw):
                continue
            cells = _target_cells(top, left, th, tw)
            for r, c in cells:
                grid[r, c] = ORANGE
            occupied |= cells
            placed = True
            break
        if not placed:
            break

    decoy_filter_count = random.randint(0, 3)
    for _ in range(decoy_filter_count):
        for _retry in range(50):
            r = random.randint(1, height - 2)
            c = random.randint(1, width - 2)
            cell = (r, c)
            if cell in reserved_path or cell in occupied:
                continue
            if any(nei in occupied for nei in _nbr4(cell)):
                continue
            grid[cell] = random.choice(palette)
            occupied.add(cell)
            break

    if not np.any(grid == GREY):
        return None
    if not np.any(grid == ORANGE):
        return None

    rotated = _rotate_scene(grid)
    try:
        out = main(rotated)
    except AssertionError:
        return None
    if out.shape != rotated.shape:
        return None
    if not np.any(out == YELLOW):
        return None
    return rotated


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-mirror ARC input")
