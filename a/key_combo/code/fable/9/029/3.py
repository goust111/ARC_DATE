import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _sorted_offsets(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _offset_patch(
    offsets: set[tuple[int, int]],
    *,
    fill_color: int,
    anchor_color: int | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    pts = set(offsets)
    if anchor_color is not None:
        pts.add((0, 0))
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    anchor = (-r0, -c0)
    if anchor_color is not None:
        patch[anchor] = int(anchor_color)
    for dr, dc in offsets:
        patch[dr - r0, dc - c0] = int(fill_color)
    return patch, anchor


def _extract_fragments(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    comps = _components(grid != bg)
    assert len(comps) >= 2, "expected multiple fragments"

    fragments = []
    for comp in comps:
        values = [int(grid[r, c]) for r, c in comp]
        gray_cells = [(r, c) for (r, c), v in zip(comp, values) if v == GREY]
        assert len(gray_cells) == 1, "each fragment must contain exactly one gray anchor"
        anchor = gray_cells[0]
        colors = {v for v in values if v not in (bg, GREY)}
        assert len(colors) == 1, "each fragment must use exactly one non-gray color"
        color = next(iter(colors))
        offsets = {
            (r - anchor[0], c - anchor[1])
            for r, c in comp
            if int(grid[r, c]) == color
        }
        assert offsets, "each fragment must contain at least one colored pixel"
        fragments.append(
            {
                "anchor": anchor,
                "color": int(color),
                "offsets": _sorted_offsets(offsets),
            }
        )
    return fragments


def _template_offsets(fragments: list[dict]) -> set[tuple[int, int]]:
    union: set[tuple[int, int]] = set()
    for fragment in fragments:
        union.update(fragment["offsets"])
    assert union, "template union cannot be empty"
    return union


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align all anchored fragments to reveal the hidden template, then complete each fragment in place.

    Input structure:
    - A black background contains several separated fragments.
    - Each fragment is one 8-connected object made of exactly one gray anchor pixel and pixels of one fragment-specific color.
    - If all fragments are translated so their gray anchors coincide, their colored pixels form one common hidden template, and no fragment is already complete.

    Rule:
    - Extract every fragment and record its colored pixels as offsets from its gray anchor.
    - Union those offsets across all fragments to recover the full template mask relative to the anchor.
    - For each fragment, repaint that full template back at the fragment's own anchor position using the fragment's color, and recolor the anchor position itself to that same color.

    Key insight:
    - The gray pixels are registration marks: only anchor-aligned overlay exposes the missing template, which is then broadcast back to every original location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    fragments = _extract_fragments(g)
    template = _template_offsets(fragments)
    assert all(set(fragment["offsets"]) < template for fragment in fragments), "no fragment may already be complete"

    full_shape = set(template)
    full_shape.add((0, 0))

    out = np.full(g.shape, bg, dtype=int)
    for fragment in fragments:
        anchor_r, anchor_c = fragment["anchor"]
        color = int(fragment["color"])
        for dr, dc in full_shape:
            rr, cc = anchor_r + dr, anchor_c + dc
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "completed shape went out of bounds"
            assert out[rr, cc] == bg, "completed shapes must not overlap"
            out[rr, cc] = color
    return out


def _path_to_root(parent: dict[tuple[int, int], tuple[int, int] | None], leaf: tuple[int, int]) -> list[tuple[int, int]]:
    path = []
    node = leaf
    while node != (0, 0):
        path.append(node)
        node = parent[node]
        assert node is not None
    path.reverse()
    return path


def _build_template(rng: random.Random) -> tuple[set[tuple[int, int]], list[set[tuple[int, int]]]]:
    for _ in range(400):
        trunk_len = rng.randint(0, 2)
        trunk_dir = rng.choice(DIRS8)
        parent: dict[tuple[int, int], tuple[int, int] | None] = {(0, 0): None}
        used = {(0, 0)}
        pos = (0, 0)

        ok = True
        for _step in range(trunk_len):
            nxt = (pos[0] + trunk_dir[0], pos[1] + trunk_dir[1])
            if max(abs(nxt[0]), abs(nxt[1])) > 3 or nxt in used:
                ok = False
                break
            parent[nxt] = pos
            used.add(nxt)
            pos = nxt
        if not ok:
            continue

        branch_root = pos
        branch_count = rng.randint(3, 5)
        first_dirs = [d for d in DIRS8 if (branch_root[0] + d[0], branch_root[1] + d[1]) not in used]
        first_dirs = [d for d in first_dirs if max(abs(branch_root[0] + d[0]), abs(branch_root[1] + d[1])) <= 3]
        if len(first_dirs) < branch_count:
            continue
        rng.shuffle(first_dirs)

        leaves: list[tuple[int, int]] = []
        for first_dir in first_dirs[:branch_count]:
            current = branch_root
            prev_dir = first_dir
            max_steps = rng.randint(1, 3)
            made_step = False
            for step_idx in range(max_steps):
                if step_idx == 0:
                    choices = [prev_dir]
                else:
                    candidates = []
                    for cand in DIRS8:
                        if cand == (-prev_dir[0], -prev_dir[1]):
                            continue
                        nxt = (current[0] + cand[0], current[1] + cand[1])
                        if nxt in used or max(abs(nxt[0]), abs(nxt[1])) > 3:
                            continue
                        weight = 4 if cand == prev_dir else 1
                        candidates.extend([cand] * weight)
                    if not candidates:
                        break
                    choices = candidates
                direction = rng.choice(choices)
                nxt = (current[0] + direction[0], current[1] + direction[1])
                if nxt in used or max(abs(nxt[0]), abs(nxt[1])) > 3:
                    if step_idx == 0:
                        made_step = False
                    break
                parent[nxt] = current
                used.add(nxt)
                current = nxt
                prev_dir = direction
                made_step = True
            if not made_step:
                ok = False
                break
            leaves.append(current)
        if not ok:
            continue

        template = set(used)
        template.remove((0, 0))
        if not (5 <= len(template) <= 10):
            continue
        rows = [r for r, _ in used]
        cols = [c for _, c in used]
        if len(set(rows)) < 2 or len(set(cols)) < 2:
            continue
        if max(rows) - min(rows) + 1 > 7 or max(cols) - min(cols) + 1 > 7:
            continue

        fragments = [set(_path_to_root(parent, leaf)) for leaf in leaves]
        union = set().union(*fragments)
        if union != template:
            continue
        if any(len(fragment) == len(template) for fragment in fragments):
            continue
        if any(not (fragment < template) for fragment in fragments):
            continue
        if any(not (fragment - set().union(*(other for other in fragments if other is not fragment))) for fragment in fragments):
            continue
        return template, fragments
    raise RuntimeError("failed to build a valid hidden template")


def _place_bboxes(
    rng: random.Random,
    count: int,
    bbox_shape: tuple[int, int],
) -> tuple[int, int, list[tuple[int, int]]]:
    box_h, box_w = bbox_shape
    min_side = max(18, int(np.ceil(np.sqrt(count))) * (max(box_h, box_w) + 2))
    min_side = min(min_side, 30)
    for _ in range(300):
        h = rng.randint(min_side, 30)
        w = rng.randint(min_side, 30)
        occupied = np.zeros((h, w), dtype=bool)
        placements: list[tuple[int, int]] = []
        ok = True
        for _idx in range(count):
            candidates = []
            for r0 in range(h - box_h + 1):
                for c0 in range(w - box_w + 1):
                    rr0 = max(0, r0 - 1)
                    cc0 = max(0, c0 - 1)
                    rr1 = min(h, r0 + box_h + 1)
                    cc1 = min(w, c0 + box_w + 1)
                    if occupied[rr0:rr1, cc0:cc1].any():
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                ok = False
                break
            choice = rng.choice(candidates)
            placements.append(choice)
            r0, c0 = choice
            occupied[r0 : r0 + box_h, c0 : c0 + box_w] = True
        if ok:
            return h, w, placements
    raise RuntimeError("failed to place fragments without overlap")


def _render_instance(
    template: set[tuple[int, int]],
    fragments: list[set[tuple[int, int]]],
    colors: list[int],
    placements: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    full_patch, full_anchor = _offset_patch(template | {(0, 0)}, fill_color=1, anchor_color=None)
    h = max(r0 for r0, _ in placements) + full_patch.shape[0]
    w = max(c0 for _, c0 in placements) + full_patch.shape[1]
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for color, frag_offsets, (r0, c0) in zip(colors, fragments, placements):
        anchor_abs = (r0 + full_anchor[0], c0 + full_anchor[1])
        frag_patch, frag_anchor = _offset_patch(frag_offsets, fill_color=color, anchor_color=GREY)
        fr0 = anchor_abs[0] - frag_anchor[0]
        fc0 = anchor_abs[1] - frag_anchor[1]
        frag_view = inp[fr0 : fr0 + frag_patch.shape[0], fc0 : fc0 + frag_patch.shape[1]]
        assert np.all(frag_view == BG), "input fragments must not overlap"
        frag_view[:, :] = np.where(frag_patch != BG, frag_patch, frag_view)

        for dr, dc in template | {(0, 0)}:
            rr, cc = anchor_abs[0] + dr, anchor_abs[1] + dc
            assert out[rr, cc] == BG, "completed shapes must stay disjoint"
            out[rr, cc] = int(color)

    return inp, out, full_patch.shape


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    template: set[tuple[int, int]],
    fragments: list[set[tuple[int, int]]],
    colors: list[int],
) -> None:
    solved = main(inp)
    assert np.array_equal(solved, expected), "solver output does not match generator target"
    assert not np.array_equal(inp, expected), "input must differ from output"

    recovered = _extract_fragments(inp)
    assert len(recovered) == len(fragments), "fragment count changed after rendering"
    expected_by_color = {
        int(color): _sorted_offsets(fragment)
        for color, fragment in zip(colors, fragments)
    }
    for fragment in recovered:
        color = int(fragment["color"])
        assert color in expected_by_color, "unexpected fragment color"
        assert tuple(fragment["offsets"]) == expected_by_color[color], "fragment geometry changed"

    recovered_template = _template_offsets(recovered)
    assert recovered_template == template, "recovered template must match the intended template"
    assert all(set(fragment["offsets"]) < template for fragment in recovered), "recovered fragment cannot be complete"
    full_area = len(template) + 1
    output_components = _components(expected != BG)
    assert len(output_components) == len(fragments), "output should contain one completed object per fragment"
    for comp in output_components:
        assert len(comp) == full_area, "every completed object must realize the full template plus anchor"


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        template, fragment_sets = _build_template(rng)
        colors = rng.sample(PALETTE, len(fragment_sets))
        full_patch, _full_anchor = _offset_patch(template | {(0, 0)}, fill_color=1, anchor_color=None)
        _h, _w, placements = _place_bboxes(rng, len(fragment_sets), full_patch.shape)
        inp, expected, _shape = _render_instance(template, fragment_sets, colors, placements)
        try:
            _validate_instance(inp, expected, template, fragment_sets, colors)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
