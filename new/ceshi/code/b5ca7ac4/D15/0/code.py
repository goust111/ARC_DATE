# Variant of b5ca7ac4
# Dimension: D15 — 干扰项——输入中的非目标对象 (src)
# Change: 输入中除框架和背景外无其他对象，所有非背景像素都属于某个框架... → 增加孤立单像素散点作为噪声

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the D15 variant
# - concretization:
#   - Target objects stay identical to the base task: 5x5 framed squares with frame color 8
#     or 2 and a uniform 3x3 interior.
#   - The new element is a small set of isolated single-pixel colored dots placed on true
#     background cells. They are not attached to any square, can reuse frame colors, and must
#     disappear in the output.
#   - Small trace example: in the base task, every non-bg pixel belongs to a square, so a solver
#     may safely treat "non-bg == object". In the variant, a teal singleton at (4, 11) is a
#     distractor: object extraction must ignore that pixel, keep only full 5x5 frames, and then
#     run the same column-order / row-overlap placement as before.
# - complexity calibration:
#   - Noise count is kept small (2-5) so the scene remains readable, but it is always present and
#     often uses frame colors 2/8 so "every red/teal pixel matters" is visibly false.
#   - Noise is still restricted to singleton dots; larger blobs would add a second reasoning
#     change unrelated to D15.
# - implicit assumptions in the original main() that D15 touches:
#   - `bg = get_bg(g)` still assumes background is dominant. The generator keeps noise sparse, so
#     background remains the most common color.
#   - The 5x5 scan assumes the meaningful structures are complete framed windows. That still
#     holds, but now there can also be stray non-bg pixels that must not influence extraction.
#   - The old code relied on "ignoring anything that is not a square" implicitly. Here that logic
#     is made explicit by deleting 8-neighborhood-singleton pixels before scanning, so the solver
#     states the new object model rather than depending on a lucky side effect.
# - placement legality:
#   - Squares remain non-overlapping.
#   - Every noise pixel must sit on a background cell whose full 3x3 neighborhood is background,
#     so it is a true singleton and cannot merge with a frame or with another noise dot.
#   - Output never contains noise; the output-first generator still validates `main(input)`.
# - visual observability / cleanliness:
#   - The change is directly visible because the input now contains extra colored specks that are
#     absent from the output.
#   - Using isolated dots avoids clutter: they are easy to see but do not obscure any square edge
#     or interior.
# - color inference:
#   - No hidden color mapping is introduced. Noise colors are arbitrary distractors; the visible
#     rule is simply that isolated dots are erased while framed squares are rearranged exactly as
#     in the source task.
# - review of the new main() assumptions:
#   - The denoiser assumes each distractor is a singleton in its 8-neighborhood.
#   - The square detector assumes target objects are complete 5x5 frames with a uniform 3x3
#     interior whose color is neither bg nor the frame color.
#   - The generator enforces both assumptions by construction.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500
NOISE_RANGE = (2, 5)


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, object]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(
                int(positions[i][0]),
                int(positions[i][1]),
                int(positions[j][0]),
                int(positions[j][1]),
            ):
                return True
    return False


def _remove_isolated_noise(grid: np.ndarray, bg: int) -> np.ndarray:
    src = np.array(grid, dtype=int)
    cleaned = src.copy()
    H, W = src.shape
    for r, c in np.argwhere(src != bg):
        r0 = max(0, int(r) - 1)
        r1 = min(H - 1, int(r) + 1)
        c0 = max(0, int(c) - 1)
        c1 = min(W - 1, int(c) + 1)
        if int(np.count_nonzero(src[r0 : r1 + 1, c0 : c1 + 1] != bg)) == 1:
            cleaned[int(r), int(c)] = bg
    return cleaned


def _extract_squares(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    H, W = grid.shape
    squares: list[dict[str, object]] = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(grid, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {AZURE, RED}:
                continue

            inner = patch[1:-1, 1:-1]
            if int(inner.min()) != int(inner.max()):
                continue
            inner_col = int(inner[0, 0])
            if inner_col in {bg, frame_col}:
                continue

            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, er) for er in edge_rows):
            edge_rows.append(int(r))
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    random.shuffle(all_rows)
    inward_rows: list[int] = []
    for r in all_rows:
        if not any(_rows_overlap(r, er) for er in edge_rows):
            continue
        if any(_rows_overlap(r, ir) for ir in inward_rows):
            continue
        inward_rows.append(int(r))
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None

    return edge_rows, inward_rows


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        patch = _make_patch(frame_color, colors[i])
        paste(output, patch, int(r), int(edge_col))
        placed.append((int(r), int(edge_col), True, patch))
    for i, r in enumerate(inward_rows):
        patch = _make_patch(frame_color, colors[len(edge_rows) + i])
        paste(output, patch, int(r), int(inward_col))
        placed.append((int(r), int(inward_col), False, patch))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]
    inward_deps = [
        [i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)]
        for (ir, _, _) in inwards
    ]

    n = len(sq_list)
    col_range = list(range(0, GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_order = list(range(len(inwards)))
        random.shuffle(inward_order)
        for iw_idx in inward_order:
            latest_pos = max(
                (
                    pos
                    for d in inward_deps[iw_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == d
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered_squares = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]
        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, patch) in enumerate(ordered_squares):
            ic = int(chosen_cols[i])
            if ic == int(oc):
                ok = False
                break
            result.append((int(r), ic, patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _can_place_noise(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    if int(grid[r, c]) != bg:
        return False
    H, W = grid.shape
    r0 = max(0, int(r) - 1)
    r1 = min(H - 1, int(r) + 1)
    c0 = max(0, int(c) - 1)
    c1 = min(W - 1, int(c) + 1)
    return not bool((grid[r0 : r1 + 1, c0 : c1 + 1] != bg).any())


def _add_noise(inp: np.ndarray, bg: int) -> bool:
    n_noise = random.randint(*NOISE_RANGE)
    extra_noise_cols = [c for c in range(10) if c not in {bg, AZURE, RED}]
    noise_palette = [AZURE, RED]
    if extra_noise_cols and random.random() < 0.8:
        noise_palette.append(random.choice(extra_noise_cols))

    forced_colors = random.sample([AZURE, RED], k=2)
    forced_colors.extend(random.choice(noise_palette) for _ in range(max(0, n_noise - 2)))

    for idx in range(n_noise):
        candidates = [
            (r, c)
            for r in range(GRID_H)
            for c in range(GRID_W)
            if _can_place_noise(inp, r, c, bg)
        ]
        if not candidates:
            return False
        r, c = random.choice(candidates)
        inp[int(r), int(c)] = int(forced_colors[idx])
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack framed 5x5 squares after removing stray singleton noise.

    Input structure:
    - bg is the dominant color in the grid.
    - Target objects are teal (8) and red (2) 5x5 frames with a uniform colored 3x3 interior.
    - Extra isolated one-cell colored dots may also sit on background cells; they are distractors.

    Rule:
    - First erase isolated non-bg singleton dots, then detect only complete 5x5 framed squares.
    - Teal squares move to the left edge and red squares move to the right edge, keeping rows.
    - Teal squares are processed by increasing input column and red squares by decreasing input
      column; row-overlap with an already edge-placed square pushes the later square one width
      inward.

    Key insight:
    - The distractors are removable clutter, so the only structures that drive the output are the
      full 5x5 frames; after filtering them, the original near-edge ordering and row-overlap logic
      still determines edge versus inward placement.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    cleaned = _remove_isolated_noise(g, bg)
    squares = _extract_squares(cleaned, bg)

    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    azure_squares = sorted(
        [s for s in squares if int(s["frame"]) == AZURE],
        key=lambda s: int(s["c0"]),
    )
    placed_at_edge: list[tuple[int, int]] = []
    for s in azure_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, np.array(s["patch"], dtype=int), r0, int(dest_c))
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    red_squares = sorted(
        [s for s in squares if int(s["frame"]) == RED],
        key=lambda s: -int(s["c0"]),
    )
    placed_at_edge = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, np.array(s["patch"], dtype=int), r0, int(dest_c))
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_output_rows(n_azure)
    if azure_assign is None:
        return None
    red_assign = _assign_output_rows(n_red)
    if red_assign is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_assign
    red_edge_rows, red_inward_rows = red_assign

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_out = _place_side_squares(
        output,
        azure_edge_rows,
        azure_inward_rows,
        azure_colors,
        AZURE,
        edge_col=0,
        inward_col=SQ,
    )
    red_out = _place_side_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )

    azure_input = _derive_input_positions(azure_out, ascending=True)
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_out, ascending=False)
    if red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in all_positions:
        paste(inp, patch, int(r), int(c))

    if not _add_noise(inp, bg):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
