from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
TERRAIN = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int, skip_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg or color in skip_colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return out


def _terrain_color(grid: np.ndarray, bg: int) -> int:
    colors = {int(v) for v in grid[-1] if int(v) != bg}
    assert len(colors) == 1, "terrain color must be the unique non-background color touching the bottom edge"
    return next(iter(colors))


def _terrain_tops(grid: np.ndarray, terrain: int) -> np.ndarray:
    h, w = grid.shape
    tops = np.full(w, h, dtype=int)
    for c in range(w):
        rows = np.where(grid[:, c] == terrain)[0]
        if rows.size:
            tops[c] = int(rows[0])
    return tops


def _component_mask_and_bottoms(component: dict) -> tuple[np.ndarray, list[int], int, int]:
    coords = component["coords"]
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    bottoms = []
    for local_c in range(mask.shape[1]):
        rows = np.where(mask[:, local_c])[0]
        assert rows.size, "each occupied object bbox column must contain object pixels"
        bottoms.append(int(rows[-1]))
    return mask, bottoms, r0, c0


def _settle_row(mask: np.ndarray, bottoms: list[int], c0: int, terrain_tops: np.ndarray) -> tuple[int, bool]:
    contact_row = min(int(terrain_tops[c0 + dc]) - 1 - bottom for dc, bottom in enumerate(bottoms))
    snug = all(contact_row + bottom == int(terrain_tops[c0 + dc]) - 1 for dc, bottom in enumerate(bottoms))
    final_row = contact_row if snug else contact_row - 1
    assert final_row >= 0, "final row must stay inside the grid"
    return final_row, snug


def _stamp(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each colored object vertically, allowing only contour-matching objects to sit flush inside the terrain.

    Input structure:
    - The background is black, and one stationary grey terrain rises from the bottom edge.
    - Above the terrain are several separated monochrome 4-connected objects with distinct column ranges.
    - The terrain columns define a jagged top contour with notches of varying depth.

    Rule:
    - Keep the terrain fixed and process each object independently as a rigid vertical drop.
    - For an object, compute the lowest non-overlapping placement over the terrain within its own columns.
    - If that first-contact placement makes every occupied object column touch the terrain directly below it, keep it there; otherwise move the object up by one row and stop.

    Key insight:
    - Pure gravity and universal one-row buffering are both wrong: only objects whose bottom contour exactly complements the terrain profile are allowed to land at the contact position.
    """
    g = np.asarray(grid, dtype=int)
    bg = BACKGROUND if np.any(g == BACKGROUND) else _get_bg(g)
    terrain = _terrain_color(g, bg)
    terrain_tops = _terrain_tops(g, terrain)

    objects = _find_components(g, bg=bg, skip_colors={terrain})
    spans = []
    for obj in objects:
        _, _, _, c0 = _component_mask_and_bottoms(obj)
        c1 = obj["bbox"][3]
        spans.append((c0, c1))
    for i, (a0, a1) in enumerate(spans):
        for b0, b1 in spans[i + 1 :]:
            assert a1 < b0 or b1 < a0, "object column ranges must not overlap"

    out = np.full_like(g, bg)
    out[g == terrain] = terrain
    for obj in objects:
        mask, bottoms, _, c0 = _component_mask_and_bottoms(obj)
        final_row, _ = _settle_row(mask, bottoms, c0, terrain_tops)
        _stamp(out, mask, final_row, c0, int(obj["color"]))
    return out


def _sample_bottom_profile(width: int, irregular: bool) -> list[int]:
    max_depth = random.randint(2, 5)
    for _ in range(80):
        bottoms = [random.randint(0, max_depth - 1) for _ in range(width)]
        if max(bottoms) == 0:
            continue
        if irregular and len(set(bottoms)) < 2:
            continue
        if not irregular and random.random() < 0.5:
            bottoms = [bottoms[0]] * width
            if max(bottoms) == 0:
                bottoms[random.randrange(width)] = 1
        return bottoms
    base = [random.randint(0, max_depth - 1) for _ in range(width)]
    base[-1] = max(base[-1], 1)
    if irregular and len(set(base)) == 1:
        base[0] = max(0, base[0] - 1)
        base[-1] = min(max_depth - 1, base[-1] + 1)
    return base


def _bottoms_to_mask(bottoms: list[int]) -> np.ndarray:
    height = max(bottoms) + 1
    mask = np.zeros((height, len(bottoms)), dtype=bool)
    for c, bottom in enumerate(bottoms):
        mask[: bottom + 1, c] = True
    return mask


def _choose_layout() -> tuple[int, list[tuple[int, int]], list[int]]:
    n_objects = random.choice([2, 3, 3])
    widths = [random.randint(2, 6) for _ in range(n_objects)]
    gaps = [random.randint(1, 3) for _ in range(n_objects + 1)]
    total = sum(widths) + sum(gaps)
    if total > 28:
        return _choose_layout()
    width = random.randint(total, min(30, total + 4))
    slack = width - total
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1
    spans = []
    c = gaps[0]
    for idx, obj_w in enumerate(widths):
        spans.append((c, c + obj_w - 1))
        c += obj_w + gaps[idx + 1]
    return width, spans, widths


def _fill_unspecified_tops(tops: list[int | None], low: int, high: int) -> list[int]:
    result = list(tops)
    for idx, value in enumerate(result):
        if value is not None:
            continue
        left = next((result[j] for j in range(idx - 1, -1, -1) if result[j] is not None), None)
        right = next((result[j] for j in range(idx + 1, len(result)) if result[j] is not None), None)
        candidates = [v for v in (left, right) if v is not None]
        anchor = int(round(sum(candidates) / len(candidates))) if candidates else random.randint(low, high)
        lo = max(low, anchor - 2)
        hi = min(high, anchor + 2)
        result[idx] = random.randint(lo, hi)
    return [int(v) for v in result]


def _render_base(shape: tuple[int, int], terrain_tops: np.ndarray) -> np.ndarray:
    base = np.full(shape, BACKGROUND, dtype=int)
    for c, top in enumerate(terrain_tops):
        base[int(top) :, c] = TERRAIN
    return base


def _build_case() -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    grid_h = random.randint(12, 19)
    grid_w, spans, widths = _choose_layout()
    terrain_tops: list[int | None] = [None] * grid_w

    branch_kinds = ["snug", "buffer"]
    if len(spans) == 3:
        branch_kinds.append(random.choice(["snug", "buffer"]))
    random.shuffle(branch_kinds)
    colors = random.sample(OBJECT_COLORS, len(spans))

    objects = []
    for idx, ((c0, c1), obj_w, branch) in enumerate(zip(spans, widths, branch_kinds)):
        irregular = branch == "buffer" or random.random() < 0.7
        bottoms = _sample_bottom_profile(obj_w, irregular=irregular)
        mask = _bottoms_to_mask(bottoms)
        max_extra = 0 if branch == "snug" else 2
        contact_min = 5
        contact_max = grid_h - mask.shape[0] - max_extra - 2
        if contact_max < contact_min:
            raise ValueError("insufficient height for sampled object")
        contact_row = random.randint(contact_min, contact_max)
        extras = [0] * obj_w
        if branch == "buffer":
            zero_cols = random.sample(range(obj_w), k=random.randint(1, max(1, obj_w - 1)))
            if len(zero_cols) == obj_w:
                zero_cols = zero_cols[:-1]
            for local_c in range(obj_w):
                extras[local_c] = 0 if local_c in zero_cols else random.randint(1, 2)
            if all(extra == 0 for extra in extras) or all(extra > 0 for extra in extras):
                raise ValueError("buffer branch must mix touching and gapped columns")
        for local_c, bottom in enumerate(bottoms):
            terrain_tops[c0 + local_c] = contact_row + 1 + bottom + extras[local_c]
        final_row = contact_row if branch == "snug" else contact_row - 1
        lift = random.randint(1, min(4, final_row))
        objects.append(
            {
                "branch": branch,
                "color": colors[idx],
                "mask": mask,
                "bottoms": bottoms,
                "left": c0,
                "final_row": final_row,
                "lift": lift,
            }
        )

    min_top = max(grid_h // 2, 6)
    max_top = grid_h - 1
    terrain_top_rows = np.array(_fill_unspecified_tops(terrain_tops, min_top, max_top), dtype=int)

    base = _render_base((grid_h, grid_w), terrain_top_rows)
    output = base.copy()
    input_grid = base.copy()
    for obj in objects:
        _stamp(output, obj["mask"], obj["final_row"], obj["left"], obj["color"])
        _stamp(input_grid, obj["mask"], obj["final_row"] - obj["lift"], obj["left"], obj["color"])
    return input_grid, output, objects, terrain_top_rows


def _semantic_check(inp: np.ndarray, out: np.ndarray, specs: list[dict], terrain_tops: np.ndarray) -> bool:
    if inp.shape != out.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if Counter(inp.ravel()).most_common(1)[0][0] != BACKGROUND:
        return False
    if len({spec["color"] for spec in specs}) != len(specs):
        return False
    if len(specs) < 2 or {spec["branch"] for spec in specs} != {"snug", "buffer"}:
        return False
    if np.array_equal(inp, out):
        return False
    if len(set(map(int, terrain_tops))) < 3:
        return False

    def render(policy: str) -> np.ndarray:
        trial = _render_base(inp.shape, terrain_tops)
        for spec in specs:
            if policy == "contact":
                row = min(int(terrain_tops[spec["left"] + dc]) - 1 - bottom for dc, bottom in enumerate(spec["bottoms"]))
            else:
                row = min(int(terrain_tops[spec["left"] + dc]) - 1 - bottom for dc, bottom in enumerate(spec["bottoms"])) - 1
            if row < 0:
                return trial
            _stamp(trial, spec["mask"], row, spec["left"], spec["color"])
        return trial

    for spec in specs:
        contact_row = min(
            int(terrain_tops[spec["left"] + dc]) - 1 - bottom
            for dc, bottom in enumerate(spec["bottoms"])
        )
        snug = all(
            contact_row + bottom == int(terrain_tops[spec["left"] + dc]) - 1
            for dc, bottom in enumerate(spec["bottoms"])
        )
        if spec["branch"] == "snug":
            if not snug or spec["final_row"] != contact_row:
                return False
        else:
            if snug or spec["final_row"] != contact_row - 1:
                return False

    if np.array_equal(render("contact"), out):
        return False
    if np.array_equal(render("buffer"), out):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, out))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, out, specs, terrain_tops = _build_case()
        except ValueError:
            continue
        if _semantic_check(inp, out, specs, terrain_tops):
            return inp
    raise RuntimeError("failed to generate a valid buffered-vs-snug terrain input")
