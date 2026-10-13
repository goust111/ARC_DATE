from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg


BG = 0
CONNECTOR = 6
BODY_COLORS = [1, 2, 3, 4, 7, 8, 9]
MIN_INPUT_SIDE = 18
MAX_INPUT_SIDE = 24
MIN_SEPARATION = 2
MAX_GENERATION_ATTEMPTS = 500


def _extract_info(grid: np.ndarray, comp: dict) -> dict:
    patch = crop_bbox(grid, comp["bbox"])
    fg = patch[patch != BG]
    color_counts = Counter(int(v) for v in fg if int(v) != CONNECTOR)
    assert len(color_counts) == 2, "each sprite must contain exactly one body color and one marker color"

    body_color, body_count = max(color_counts.items(), key=lambda kv: kv[1])
    marker_colors = [color for color, count in color_counts.items() if color != body_color and count == 1]
    assert len(marker_colors) == 1, "expected exactly one embedded marker pixel"
    marker_color = int(marker_colors[0])
    assert int(body_count) > 1, "body color must be the majority color"

    connector_pts = np.argwhere(patch == CONNECTOR)
    assert connector_pts.shape == (1, 2), "expected exactly one magenta connector pixel"
    connector = tuple(map(int, connector_pts[0]))

    return {
        "patch": patch,
        "body_color": int(body_color),
        "marker_color": marker_color,
        "connector": connector,
    }


def _order_chain(infos: list[dict]) -> list[dict]:
    by_body: dict[int, dict] = {}
    referenced: set[int] = set()
    for info in infos:
        body = int(info["body_color"])
        assert body not in by_body, "body colors must be unique across sprites"
        by_body[body] = info
        referenced.add(int(info["marker_color"]))

    heads = [info for info in infos if int(info["body_color"]) not in referenced]
    assert len(heads) == 1, "the color-pairing graph must have exactly one head"

    ordered: list[dict] = []
    used_bodies: set[int] = set()
    cur = heads[0]
    while True:
        body = int(cur["body_color"])
        assert body not in used_bodies, "expected a linear chain, not a cycle"
        ordered.append(cur)
        used_bodies.add(body)

        nxt = by_body.get(int(cur["marker_color"]))
        if nxt is None:
            break
        cur = nxt

    assert len(ordered) == len(infos), "chain traversal must visit every sprite exactly once"
    return ordered


def _assemble_chain(ordered: list[dict], *, recolor_connectors: bool) -> np.ndarray:
    assert ordered, "expected at least one sprite"

    positions: list[tuple[int, int]] = []
    align_row = int(ordered[0]["connector"][0])

    for idx, info in enumerate(ordered):
        patch = info["patch"]
        _h, w = map(int, patch.shape)
        cr, cc = map(int, info["connector"])

        if idx == 0:
            assert cc == w - 1, "the chain head must expose its connector on the right edge"
            x = 0
        else:
            assert cc == 0, "every non-head sprite must expose its connector on the left edge"
            prev = ordered[idx - 1]
            prev_patch = prev["patch"]
            prev_w = int(prev_patch.shape[1])
            prev_cc = int(prev["connector"][1])
            assert prev_cc == prev_w - 1, "every predecessor must expose its connector on the right edge"
            prev_x, _prev_y = positions[idx - 1]
            x = int(prev_x + prev_cc + 1 - cc)

        y = int(align_row - cr)
        positions.append((int(x), int(y)))

    min_r = min(y for _x, y in positions)
    min_c = min(c for c, _r in positions)
    max_r = max(int(y) + int(info["patch"].shape[0]) - 1 for info, (_x, y) in zip(ordered, positions))
    max_c = max(int(x) + int(info["patch"].shape[1]) - 1 for info, (x, _y) in zip(ordered, positions))

    out = np.full((int(max_r - min_r + 1), int(max_c - min_c + 1)), BG, dtype=int)
    for info, (x, y) in zip(ordered, positions):
        patch = info["patch"].copy()
        if recolor_connectors:
            patch[patch == CONNECTOR] = int(info["body_color"])

        h, w = map(int, patch.shape)
        r0 = int(y - min_r)
        c0 = int(x - min_c)
        window = out[r0 : r0 + h, c0 : c0 + w]
        mask = patch != BG
        assert not np.any(window[mask] != BG), "assembled sprites must not overlap"
        window[mask] = patch[mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Order marked multicolor sprites into a color-paired chain, dock them into one strip, and absorb the magenta connectors into each body color.

    Input structure:
    - The background is black.
    - The foreground consists of 4-connected multicolor sprites that are separated from one another.
    - Each sprite has one majority body color, one embedded marker pixel whose color names the next sprite's body color, and one magenta connector pixel on the left or right bbox edge.

    Rule:
    - For each sprite, identify the body color as the most frequent non-black, non-magenta color, and identify the unique remaining non-black, non-magenta pixel as the marker color.
    - Find the unique chain head whose body color is not referenced by any other marker; then follow marker-color-to-body-color matches to order all sprites into a single chain.
    - Crop each sprite to its bbox and place the head at the left; each later sprite is translated so its left-edge connector sits immediately to the right of the previous sprite's right-edge connector on the same row.
    - Recolor every magenta connector pixel to its owning sprite's body color and return the minimal crop containing the assembled strip.

    Key insight:
    - The ordering cue is not geometric proximity or edge-shape matching: the embedded marker color inside one sprite names the body color of the next sprite, and that color graph uniquely determines the docking order.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG, "this task uses black as the background"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert len(components) >= 2, "expected multiple sprites"

    infos = [_extract_info(g, comp) for comp in components]
    ordered = _order_chain(infos)
    return _assemble_chain(ordered, recolor_connectors=True)


def _connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            nxt = (int(nr), int(nc))
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _cells_to_patch(
    cells: set[tuple[int, int]],
    *,
    connector: tuple[int, int],
    body_color: int,
    marker_color: int,
) -> np.ndarray:
    assert connector in cells
    assert _connected(cells)

    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in cells:
        patch[int(r), int(c)] = int(body_color)

    marker_choices = [cell for cell in cells if cell != connector]
    marker = random.choice(marker_choices)
    patch[int(marker[0]), int(marker[1])] = int(marker_color)
    patch[int(connector[0]), int(connector[1])] = CONNECTOR
    return patch


def _make_line_patch(body_color: int, marker_color: int, row_off: int) -> np.ndarray:
    height = random.randint(max(5, row_off + 3), 8)
    cells = {(r, 0) for r in range(height)}
    connector = (int(row_off), 0)
    return _cells_to_patch(cells, connector=connector, body_color=body_color, marker_color=marker_color)


def _make_end_patch(side: str, body_color: int, marker_color: int, row_off: int) -> np.ndarray:
    assert side in {"left", "right"}

    for _ in range(120):
        width = random.choice([2, 2, 3])
        min_h = max(3, row_off + 2)
        max_h = 7 if width == 2 else 6
        if min_h > max_h:
            continue
        height = random.randint(min_h, max_h)

        if side == "right":
            cells = {(r, 0) for r in range(height)}
            cells.update((row_off, c) for c in range(1, width))
            connector = (int(row_off), int(width - 1))
            branch_col = 1
        else:
            cells = {(r, width - 1) for r in range(height)}
            cells.update((row_off, c) for c in range(width - 1))
            connector = (int(row_off), 0)
            branch_col = width - 2

        candidates = [(r, branch_col) for r in range(height) if r != row_off]
        random.shuffle(candidates)
        max_extra = min(2, 8 - len(cells))
        if max_extra > 0:
            extra_count = random.randint(0, max_extra)
            cells.update(candidates[:extra_count])

        if 5 <= len(cells) <= 8 and _connected(cells):
            return _cells_to_patch(
                cells,
                connector=connector,
                body_color=body_color,
                marker_color=marker_color,
            )

    raise RuntimeError("failed to build an end sprite")


def _fg_cells(patch: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(patch != BG)]


def _can_place(
    patch: np.ndarray,
    r0: int,
    c0: int,
    shape: tuple[int, int],
    occupied: list[tuple[int, int]],
) -> bool:
    h, w = map(int, patch.shape)
    H, W = map(int, shape)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    for pr, pc in _fg_cells(patch):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        for orr, occ in occupied:
            if max(abs(rr - int(orr)), abs(cc - int(occ))) <= MIN_SEPARATION:
                return False
    return True


def _scatter_patches(patches: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray | None:
    H, W = map(int, shape)
    grid = np.full((H, W), BG, dtype=int)
    occupied: list[tuple[int, int]] = []

    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch = patches[idx]
        h, w = map(int, patch.shape)

        placed = False
        for _ in range(300):
            r0 = random.randint(0, H - h)
            c0 = random.randint(0, W - w)
            if not _can_place(patch, r0, c0, shape, occupied):
                continue

            for pr, pc in _fg_cells(patch):
                rr = int(r0 + pr)
                cc = int(c0 + pc)
                grid[rr, cc] = int(patch[pr, pc])
                occupied.append((rr, cc))
            placed = True
            break

        if not placed:
            return None

    return grid


def _build_chain_patches() -> list[np.ndarray]:
    n = random.choice([4, 5])
    chain_bodies = random.sample(BODY_COLORS, n)
    tail_marker_choices = [c for c in BODY_COLORS if c not in chain_bodies]
    tail_marker = random.choice(tail_marker_choices)
    row_off = random.choice([0, 1, 2])

    patches: list[np.ndarray] = []
    for idx, body_color in enumerate(chain_bodies):
        marker_color = chain_bodies[idx + 1] if idx < n - 1 else tail_marker
        if idx == 0:
            patch = _make_end_patch("right", body_color, marker_color, row_off)
        elif idx == n - 1:
            patch = _make_end_patch("left", body_color, marker_color, row_off)
        else:
            patch = _make_line_patch(body_color, marker_color, row_off)
        patches.append(patch)

    return patches


def _expected_output_from_patches(patches: list[np.ndarray]) -> np.ndarray:
    infos = []
    for patch in patches:
        fake = np.zeros_like(patch)
        fake[:, :] = patch
        comp = {
            "bbox": (0, 0, int(patch.shape[0]) - 1, int(patch.shape[1]) - 1),
        }
        infos.append(_extract_info(fake, comp))
    ordered = _order_chain(infos)
    return _assemble_chain(ordered, recolor_connectors=True)


def _gen_attempt() -> np.ndarray | None:
    patches = _build_chain_patches()
    shape = (
        random.randint(MIN_INPUT_SIDE, MAX_INPUT_SIDE),
        random.randint(MIN_INPUT_SIDE, MAX_INPUT_SIDE),
    )
    grid = _scatter_patches(patches, shape)
    if grid is None:
        return None

    expected = _expected_output_from_patches(patches)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-docking input")
