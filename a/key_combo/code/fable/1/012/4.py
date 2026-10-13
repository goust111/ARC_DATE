from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SIDES = ("up", "down", "left", "right")
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = tuple(range(1, 10))

FRAGMENT_COUNT_RANGE = (3, 5)
FRAGMENT_SIZE_RANGE = (3, 5)
INPUT_SIZE_RANGE = (16, 30)
GEN_RETRIES = 600
SCATTER_RETRIES = 160


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _marker_side(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = pos
    r0, c0, r1, c1 = bbox
    sides = []
    if r == r0:
        sides.append("up")
    if r == r1:
        sides.append("down")
    if c == c0:
        sides.append("left")
    if c == c1:
        sides.append("right")
    assert len(sides) == 1, "marker must lie on exactly one outer edge"
    return sides[0]


def _other_marker_index(markers: list[dict], idx: int) -> int | None:
    if len(markers) == 1:
        return None
    assert len(markers) == 2
    return 1 - int(idx)


def _fragment_cells(fragment: dict, bg: int) -> list[tuple[int, int, int]]:
    patch = fragment["patch"]
    coords = np.argwhere(patch != bg)
    return [(int(r), int(c), int(patch[r, c])) for r, c in coords]


def _parse_fragments(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fragments = []
    for coords in _components_mask(g != bg):
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(g, bbox)
        values = g[coords[:, 0], coords[:, 1]]
        counts = Counter(map(int, values))
        body_color, body_count = counts.most_common(1)[0]
        marker_items = [(color, count) for color, count in counts.items() if int(color) != int(body_color)]
        assert 1 <= len(marker_items) <= 2, "each fragment needs one or two marker colors"
        assert body_count > max(count for _color, count in marker_items), "body color must dominate the fragment"

        markers = []
        for color, count in sorted(marker_items):
            assert int(count) == 1, "markers must be singleton pixels"
            pos_abs = coords[values == int(color)][0]
            abs_rc = (int(pos_abs[0]), int(pos_abs[1]))
            side = _marker_side(abs_rc, bbox)
            markers.append(
                {
                    "color": int(color),
                    "local": (int(abs_rc[0] - bbox[0]), int(abs_rc[1] - bbox[1])),
                    "side": side,
                }
            )

        fragments.append({"patch": patch, "bbox": bbox, "body_color": int(body_color), "markers": markers})
    assert len(fragments) >= 2, "expected at least two fragments"
    return bg, fragments


def _color_map(fragments: list[dict]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for fi, fragment in enumerate(fragments):
        for mi, marker in enumerate(fragment["markers"]):
            out.setdefault(int(marker["color"]), []).append((fi, mi))
    for color, occs in out.items():
        assert len(occs) == 2, f"marker color {color} must appear exactly twice"
    return out


def _walk_chain(fragments: list[dict], start_idx: int) -> list[tuple[int, int | None, int | None]]:
    assert len(fragments[start_idx]["markers"]) == 1, "chain must start at a single-marker endpoint"
    color_links = _color_map(fragments)

    visited = set()
    chain: list[tuple[int, int | None, int | None]] = []
    cur_idx = int(start_idx)
    entry_idx: int | None = None
    exit_idx: int | None = 0

    while True:
        if cur_idx in visited:
            raise AssertionError("chain revisits a fragment")
        visited.add(cur_idx)
        chain.append((cur_idx, entry_idx, exit_idx))
        if exit_idx is None:
            break

        exit_marker = fragments[cur_idx]["markers"][exit_idx]
        next_occ = [occ for occ in color_links[int(exit_marker["color"])] if occ != (cur_idx, exit_idx)]
        assert len(next_occ) == 1
        next_idx, next_marker_idx = next_occ[0]
        if next_idx in visited:
            raise AssertionError("marker chain closes early")

        next_markers = fragments[next_idx]["markers"]
        if len(next_markers) == 1:
            assert next_marker_idx == 0, "tail marker index mismatch"
            entry_idx = 0
            exit_idx = None
        else:
            entry_idx = int(next_marker_idx)
            exit_idx = _other_marker_index(next_markers, next_marker_idx)
            assert exit_idx is not None
        cur_idx = int(next_idx)

    assert len(visited) == len(fragments), "chain must cover every fragment"
    assert len(chain[-1]) == 3 and chain[-1][2] is None, "chain must end at the tail"
    return chain


def _assemble_chain(fragments: list[dict], chain: list[tuple[int, int | None, int | None]], bg: int) -> np.ndarray:
    placements: dict[int, tuple[int, int]] = {}
    occupied: dict[tuple[int, int], int] = {}

    for step, (frag_idx, entry_idx, exit_idx) in enumerate(chain):
        fragment = fragments[frag_idx]
        if step == 0:
            placements[frag_idx] = (0, 0)
        else:
            prev_idx, _prev_entry, prev_exit_idx = chain[step - 1]
            assert prev_exit_idx is not None
            prev_marker = fragments[prev_idx]["markers"][prev_exit_idx]
            dr, dc = DIRS[prev_marker["side"]]
            prev_top, prev_left = placements[prev_idx]
            prev_r = prev_top + int(prev_marker["local"][0])
            prev_c = prev_left + int(prev_marker["local"][1])

            assert entry_idx is not None
            cur_marker = fragment["markers"][entry_idx]
            placements[frag_idx] = (
                int(prev_r + dr - int(cur_marker["local"][0])),
                int(prev_c + dc - int(cur_marker["local"][1])),
            )

        top, left = placements[frag_idx]
        for r, c, value in _fragment_cells(fragment, bg):
            abs_rc = (int(top + r), int(left + c))
            if abs_rc in occupied:
                raise AssertionError("assembled fragments overlap")
            occupied[abs_rc] = int(value)

    coords = np.array(list(occupied.keys()), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for (r, c), value in occupied.items():
        out[int(r - r0), int(c - c0)] = int(value)
    return out


def _candidate_outputs_from_fragments(fragments: list[dict], bg: int) -> list[np.ndarray]:
    endpoints = [i for i, fragment in enumerate(fragments) if len(fragment["markers"]) == 1]
    assert len(endpoints) == 2, "expected exactly two single-marker endpoints"
    outputs: list[np.ndarray] = []
    for start_idx in endpoints:
        try:
            chain = _walk_chain(fragments, start_idx)
            out = _assemble_chain(fragments, chain, bg)
        except AssertionError:
            continue
        if not any(np.array_equal(out, existing) for existing in outputs):
            outputs.append(out)
    return outputs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered fragments by following the hidden color-pointer chain and crop the completed snake.

    Input structure:
    - A black background contains several separated connected fragments.
    - Each fragment has one dominant body color plus one or two single-pixel marker colors on its outer edge.
    - Exactly two fragments have only one marker; all other fragments have two markers.

    Rule:
    - Treat each marker color as a pointer that appears exactly twice in the whole grid.
    - Starting from the unique valid endpoint orientation, follow matching marker colors to recover the fragment chain.
    - Keep every fragment unrotated and untranslated only by sliding it so the next fragment's matched marker becomes orthogonally adjacent in the outward direction of the current fragment's edge marker.
    - After every fragment is placed, return the minimal bounding-box crop of the assembled result.

    Key insight:
    - The original scatter layout is irrelevant: chain order comes from repeated marker colors, while the edge containing each current marker fixes the direction of the next attachment.
    """
    bg, fragments = _parse_fragments(np.asarray(grid, dtype=int))
    outputs = _candidate_outputs_from_fragments(fragments, bg)
    assert len(outputs) == 1, "expected a unique non-overlapping assembly"
    return outputs[0]


def _random_edge_cell(side: str, h: int, w: int) -> tuple[int, int]:
    assert h >= 3 and w >= 3
    if side == "up":
        return 0, random.randint(1, w - 2)
    if side == "down":
        return h - 1, random.randint(1, w - 2)
    if side == "left":
        return random.randint(1, h - 2), 0
    if side == "right":
        return random.randint(1, h - 2), w - 1
    raise ValueError(side)


def _make_fragment(
    h: int,
    w: int,
    body_color: int,
    *,
    entry_color: int | None,
    exit_color: int | None,
    entry_side: str | None,
    exit_side: str | None,
) -> dict:
    patch = np.full((h, w), int(body_color), dtype=int)
    markers = []
    used = set()

    if entry_color is not None:
        assert entry_side is not None
        entry_pos = _random_edge_cell(entry_side, h, w)
        used.add(entry_pos)
        patch[entry_pos] = int(entry_color)
        markers.append({"color": int(entry_color), "local": entry_pos, "side": entry_side})

    if exit_color is not None:
        assert exit_side is not None
        exit_pos = _random_edge_cell(exit_side, h, w)
        while exit_pos in used:
            exit_pos = _random_edge_cell(exit_side, h, w)
        patch[exit_pos] = int(exit_color)
        markers.append({"color": int(exit_color), "local": exit_pos, "side": exit_side})

    entry_idx = None if entry_color is None else 0
    if exit_color is None:
        exit_idx = None
    elif entry_color is None:
        exit_idx = 0
    else:
        exit_idx = 1

    return {
        "patch": patch,
        "markers": markers,
        "true_entry": entry_idx,
        "true_exit": exit_idx,
    }


def _try_build_true_chain() -> tuple[list[dict], np.ndarray] | None:
    n_fragments = random.randint(*FRAGMENT_COUNT_RANGE)
    picked = random.sample(PALETTE, 2 * n_fragments - 1)
    body_colors = picked[:n_fragments]
    seam_colors = picked[n_fragments:]

    fragments: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    bbox: tuple[int, int, int, int] | None = None
    exit_sides: list[str] = []
    placements: list[tuple[int, int]] = []

    for idx in range(n_fragments):
        for _ in range(70):
            h = random.randint(*FRAGMENT_SIZE_RANGE)
            w = random.randint(*FRAGMENT_SIZE_RANGE)

            if idx == 0:
                entry_color = None
                entry_side = None
                exit_color = seam_colors[0]
                exit_side = random.choice(SIDES)
            elif idx == n_fragments - 1:
                entry_color = seam_colors[idx - 1]
                entry_side = random.choice(SIDES)
                exit_color = None
                exit_side = None
            else:
                entry_color = seam_colors[idx - 1]
                exit_color = seam_colors[idx]
                entry_side = random.choice(SIDES)
                exit_side = random.choice([side for side in SIDES if side != entry_side])

            fragment = _make_fragment(
                h,
                w,
                body_colors[idx],
                entry_color=entry_color,
                exit_color=exit_color,
                entry_side=entry_side,
                exit_side=exit_side,
            )

            if idx == 0:
                top_left = (0, 0)
            else:
                prev_fragment = fragments[-1]
                prev_top, prev_left = placements[-1]
                prev_exit = prev_fragment["markers"][prev_fragment["true_exit"]]
                dr, dc = DIRS[prev_exit["side"]]
                target_r = prev_top + int(prev_exit["local"][0]) + dr
                target_c = prev_left + int(prev_exit["local"][1]) + dc
                cur_entry = fragment["markers"][fragment["true_entry"]]
                top_left = (
                    int(target_r - int(cur_entry["local"][0])),
                    int(target_c - int(cur_entry["local"][1])),
                )

            local_cells = _fragment_cells(fragment, BG)
            cells = {(int(top_left[0] + r), int(top_left[1] + c)) for r, c, _value in local_cells}
            if occupied & cells:
                continue

            coords = np.array(list(cells), dtype=int)
            cur_bbox = _bbox_coords(coords)
            next_bbox = cur_bbox if bbox is None else (
                min(bbox[0], cur_bbox[0]),
                min(bbox[1], cur_bbox[1]),
                max(bbox[2], cur_bbox[2]),
                max(bbox[3], cur_bbox[3]),
            )
            if next_bbox[2] - next_bbox[0] + 1 > 30 or next_bbox[3] - next_bbox[1] + 1 > 30:
                continue

            fragments.append(fragment)
            placements.append(top_left)
            occupied |= cells
            bbox = next_bbox
            if fragment["true_exit"] is not None:
                exit_sides.append(fragment["markers"][fragment["true_exit"]]["side"])
            break
        else:
            return None

    if len(set(exit_sides)) < 2:
        return None

    solver_fragments = [{"patch": f["patch"], "markers": f["markers"]} for f in fragments]
    outputs = _candidate_outputs_from_fragments(solver_fragments, BG)
    if len(outputs) != 1:
        return None
    expected = outputs[0]
    if max(expected.shape) > 30:
        return None
    return solver_fragments, expected


def _boxes_are_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _placements_are_decoy(placements: list[tuple[int, int, int, int, int]]) -> bool:
    n = len(placements)
    forward = list(range(n))
    reverse = list(reversed(forward))

    orders = [
        [idx for idx, _r, _c, _h, _w in sorted(placements, key=lambda item: (item[1], item[2]))],
        [idx for idx, _r, _c, _h, _w in sorted(placements, key=lambda item: (item[2], item[1]))],
        [idx for idx, _r, _c, _h, _w in sorted(placements, key=lambda item: item[2])],
        [idx for idx, _r, _c, _h, _w in sorted(placements, key=lambda item: item[1])],
    ]
    return all(order not in (forward, reverse) for order in orders)


def _scatter_fragments(fragments: list[dict]) -> np.ndarray | None:
    dims = [(fragment["patch"].shape[0], fragment["patch"].shape[1]) for fragment in fragments]
    min_h = max(INPUT_SIZE_RANGE[0], max(h for h, _w in dims) + 2)
    min_w = max(INPUT_SIZE_RANGE[0], max(w for _h, w in dims) + 2)

    for _ in range(SCATTER_RETRIES):
        h = random.randint(min_h, INPUT_SIZE_RANGE[1])
        w = random.randint(min_w, INPUT_SIZE_RANGE[1])
        placements = []

        order = list(range(len(fragments)))
        random.shuffle(order)
        success = True
        for idx in order:
            ph, pw = dims[idx]
            candidates = []
            for r0 in range(h - ph + 1):
                for c0 in range(w - pw + 1):
                    box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                    if all(_boxes_are_separated(box, other_box) for _j, *other_box in placements):
                        candidates.append((r0, c0))
            if not candidates:
                success = False
                break
            r0, c0 = random.choice(candidates)
            placements.append((idx, r0, c0, r0 + ph - 1, c0 + pw - 1))

        if not success or not _placements_are_decoy(placements):
            continue

        grid = np.full((h, w), BG, dtype=int)
        for idx, r0, c0, _r1, _c1 in placements:
            patch = fragments[idx]["patch"]
            grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _try_build_true_chain()
        if built is None:
            continue
        fragments, expected = built
        grid = _scatter_fragments(fragments)
        if grid is None:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
