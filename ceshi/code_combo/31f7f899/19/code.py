# Combo variant of 31f7f899 (index 19)
# combo_spec: D1/6+D3/2
# Name: 连接器匹配+T形条

from __future__ import annotations

"""
Design analysis for the combo variant "连接器匹配+T形条".

a) Concrete behavior and trace
   - The sortable objects are no longer plain rectangles. Each movable object is a monochrome
     T shape above the base line, plus a small connector signature below the base line.
   - The base line still spans the full grid width, so the source family remains visually tied to
     31f7f899. The change is in the matching rule: objects are not ordered by height at all.
   - Each target slot is indicated by a small colored connector symbol placed below the base line.
     The left-to-right slot anchors stay fixed, and the slot colors are preserved in the output.
   - The solver must:
       1. detect each T-shaped object touching the base line;
       2. read the connector signature from the pixels below the base line;
       3. detect each slot symbol below the base line and read its signature;
       4. move each full T+connector silhouette to the slot whose signature matches it;
       5. recolor the moved silhouette to the slot color.
   - Example:
       left slot symbol = right-hook, middle = tee, right = left-hook
       input objects by signature left->right = tee, left-hook, right-hook
       output objects left->right = right-hook, tee, left-hook
     The visible control variable is local connector shape, not height.

b) Complexity calibration
   - The connector alphabet is restricted to three clearly distinct connected shapes:
     left-hook, tee, right-hook. This keeps the matching ARC-like and easy to inspect.
   - Each sample uses exactly three objects and all three connector types exactly once, preventing
     ambiguity in the matching relation.
   - T arms vary between width 3 and 5, and stem heights vary independently. Those geometric
     differences make the objects look meaningfully non-rectangular without taking over the task.

c) Audit of implicit assumptions changed by the combo
   - The source solver assumed the decisive attribute was bbox height. That assumption is broken:
     here height is only distractor information.
   - The anchor-from-base-row logic still works because every T stem crosses the base row at one
     central pixel.
   - The original code treated every non-base component as a movable bar. That no longer holds:
     slot symbols are also non-base components, but they never touch the base row. The solver now
     separates "objects touching the base row" from "slot symbols below the base row".
   - Matching now depends on connector signatures being unique and readable. The generator
     guarantees this by using all three signatures exactly once and by keeping slot symbols
     vertically separated from object connectors.

d) Placement legality
   - Objects are placed in reserved windows so that a width-5 T at any slot stays in bounds and
     never overlaps a neighbor.
   - Slot symbols live two rows below the object connectors, so the input always exposes both the
     connector and the target signature independently.
   - Each generated sample is output-first validated by checking `main(input) == output`.

e) Visual observability
   - Across examples, a human sees T-shaped objects with different connector tails and small slot
     symbols below them. The output consistently relocates objects according to connector shape.
   - The generator rejects samples whose output would also be explained by the source task's
     height-sorting rule, so the connector cue is necessary rather than decorative.

New implicit assumptions after the change
   - `main()` assumes every movable object touches the base row and has at least one connector
     pixel below it.
   - `generate_input()` guarantees that by constructing each object from a T stem plus one of the
     three connector signatures, and by placing slot symbols only below the base row.
"""

import numpy as np
from arc_utils import bbox_h, find_components, get_bg


def _norm_signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


SIG_LEFT = _norm_signature([(1, 0), (2, 0), (2, -1)])
SIG_TEE = _norm_signature([(1, -1), (1, 0), (1, 1)])
SIG_RIGHT = _norm_signature([(1, 0), (2, 0), (2, 1)])
SIGNATURES = (SIG_LEFT, SIG_TEE, SIG_RIGHT)


def _slot_signature_from_component(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    top_r = min(r for r, _c in coords)
    top_cols = sorted(c for r, c in coords if r == top_r)
    anchor = int(top_cols[len(top_cols) // 2])
    return _norm_signature([(r - top_r + 1, c - anchor) for r, c in coords])


def _find_base(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))
    best = None
    for r in range(h):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best
    return bg, int(base_r), int(base_col)


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, object]], list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, base_r, base_col = _find_base(g)

    objects: list[dict[str, object]] = []
    slots: list[dict[str, object]] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in {bg, base_col}:
            continue

        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        rows = {r for r, _c in coords}

        if base_r in rows:
            base_cols = sorted(c for r, c in coords if r == base_r)
            assert base_cols, "object must cross the base row"
            anchor = int(base_cols[len(base_cols) // 2])
            rel = [(r - base_r, c - anchor) for r, c in coords]
            connector = _norm_signature([(dr, dc) for dr, dc in rel if dr > 0])
            assert connector in SIGNATURES, "unknown connector signature"
            objects.append(
                {
                    "color": color,
                    "anchor": anchor,
                    "height": int(bbox_h(comp["bbox"])),
                    "signature": connector,
                    "rel": rel,
                }
            )
        elif min(rows) > base_r:
            top_r = min(r for r, _c in coords)
            top_cols = sorted(c for r, c in coords if r == top_r)
            anchor = int(top_cols[len(top_cols) // 2])
            signature = _slot_signature_from_component(coords)
            assert signature in SIGNATURES, "unknown slot signature"
            rel = [(r - top_r + 1, c - anchor) for r, c in coords]
            slots.append(
                {
                    "color": color,
                    "anchor": anchor,
                    "signature": signature,
                    "rel": rel,
                }
            )

    return bg, int(base_r), int(base_col), objects, slots


def _render_slot_symbols(out: np.ndarray, base_r: int, slots: list[dict[str, object]]) -> None:
    for slot in slots:
        color = int(slot["color"])
        anchor = int(slot["anchor"])
        for dr, dc in slot["rel"]:
            rr = int(base_r) + int(dr) + 2
            cc = int(anchor) + int(dc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match T-shaped objects to slots by connector signature instead of sorting by height.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, base_r, base_col, objects, slots = _extract_scene(g)

    slot_by_sig = {slot["signature"]: slot for slot in slots}
    assert len(slot_by_sig) == len(slots) == len(objects)

    out = np.full((h, w), bg, dtype=int)
    out[base_r, :] = int(base_col)
    _render_slot_symbols(out, base_r, slots)

    for obj in objects:
        slot = slot_by_sig[obj["signature"]]
        slot_color = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = slot_color

    return out


def _height_reference(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, base_r, base_col, objects, slots = _extract_scene(g)

    slots_sorted = sorted(slots, key=lambda s: (int(s["anchor"]), int(s["color"])))
    shapes_sorted = sorted(objects, key=lambda o: (int(o["height"]), int(o["anchor"]), int(o["color"])))

    out = np.full((h, w), bg, dtype=int)
    out[base_r, :] = int(base_col)
    _render_slot_symbols(out, base_r, slots)

    for slot, obj in zip(slots_sorted, shapes_sorted):
        slot_color = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = slot_color

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(n: int, k: int, slot_w: int) -> list[int]:
    used = int(k * slot_w + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("slots do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(slot_w)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _make_object_rel(stem_up: int, arm_width: int, signature: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    top_r = -int(stem_up)
    half = int(arm_width) // 2
    for dc in range(-half, half + 1):
        pts.add((top_r, dc))
    for rr in range(top_r, 1):
        pts.add((rr, 0))
    for dr, dc in signature:
        pts.add((int(dr), int(dc)))
    return sorted(pts)


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    slot_colors: list[int],
    slot_sigs: list[tuple[tuple[int, int], ...]],
    objects: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)

    for anchor, slot_color, slot_sig in zip(anchors, slot_colors, slot_sigs):
        for dr, dc in slot_sig:
            rr = int(base_r) + int(dr) + 2
            cc = int(anchor) + int(dc)
            grid[rr, cc] = int(slot_color)

    for anchor, obj in zip(anchors, objects):
        color = int(obj["color"])
        for dr, dc in obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(anchor) + int(dc)
            grid[rr, cc] = color

    return grid


def _sample_object_specs() -> list[dict[str, object]] | None:
    stem_choices = [4, 5, 6]
    arm_choices = [3, 5]

    for _ in range(200):
        sigs = list(SIGNATURES)
        np.random.shuffle(sigs)
        stems = [int(x) for x in np.random.choice(stem_choices, size=3, replace=False)]
        arms = [int(x) for x in np.random.choice(arm_choices, size=3, replace=True)]
        if len(set(arms)) < 2:
            continue

        specs = []
        for sig, stem_up, arm_width in zip(sigs, stems, arms):
            rel = _make_object_rel(stem_up, arm_width, sig)
            specs.append(
                {
                    "signature": sig,
                    "stem_up": int(stem_up),
                    "arm_width": int(arm_width),
                    "height": int(max(r for r, _c in rel) - min(r for r, _c in rel) + 1),
                    "rel": rel,
                }
            )

        by_sig = {spec["signature"]: spec for spec in specs}
        sig_order = list(by_sig)
        sig_order.sort(key=lambda s: (by_sig[s]["height"], by_sig[s]["arm_width"]))
        if sig_order == list(SIGNATURES):
            continue
        return specs
    return None


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice([19, 21, 23]))
    base_r = n // 2
    bg = 8
    base_color = 6
    k = 3
    slot_w = 5

    lefts = _slot_lefts(n, k, slot_w)
    anchors = [int(left + slot_w // 2) for left in lefts]

    palette = [1, 2, 3, 4, 5, 7, 9]
    colors = [int(c) for c in np.random.choice(palette, size=6, replace=False)]
    object_colors = colors[:k]
    slot_colors = colors[k:]

    specs = _sample_object_specs()
    if specs is None:
        return None
    by_sig = {spec["signature"]: spec for spec in specs}

    slot_sigs = list(SIGNATURES)
    np.random.shuffle(slot_sigs)

    out_objects = []
    for slot_sig, slot_color in zip(slot_sigs, slot_colors):
        spec = dict(by_sig[slot_sig])
        spec["color"] = int(slot_color)
        out_objects.append(spec)

    input_perm = np.random.permutation(k)
    if np.array_equal(input_perm, np.arange(k)):
        input_perm = np.array([1, 2, 0])
    in_objects = []
    for idx, obj_idx in enumerate(input_perm):
        spec = dict(out_objects[int(obj_idx)])
        spec["color"] = int(object_colors[idx])
        in_objects.append(spec)

    out = _draw_scene(n, base_r, base_color, bg, anchors, slot_colors, slot_sigs, out_objects)
    inp = _draw_scene(n, base_r, base_color, bg, anchors, slot_colors, slot_sigs, in_objects)

    if np.array_equal(inp, out):
        return None
    if np.array_equal(main(inp), _height_reference(inp)):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
