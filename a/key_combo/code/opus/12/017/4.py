from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
SEP = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_frame_component(comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        local[r - r0, c - c0] = True
    return np.array_equal(local, _frame_mask(h, w))


def _find_separator(grid: np.ndarray) -> int:
    H, W = grid.shape
    candidates = [c for c in range(1, W - 1) if np.all(grid[:, c] == SEP)]
    assert len(candidates) == 1, f"expected one full separator column, got {candidates}"
    return int(candidates[0])


def _extract_frames(grid: np.ndarray):
    sep = _find_separator(grid)
    frames = []
    frame_cells: set[tuple[int, int]] = set()
    for color in sorted(set(map(int, grid.flatten())) - {BG, SEP}):
        for comp in _components_of_color(grid, color):
            if not _is_frame_component(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            side = "left" if c1 < sep else "right"
            assert side in {"left", "right"}
            frame = {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "shape": (int(r1 - r0 + 1), int(c1 - c0 + 1)),
                "side": side,
            }
            frames.append(frame)
            frame_cells.update(comp)

    left = [f for f in frames if f["side"] == "left"]
    right = [f for f in frames if f["side"] == "right"]
    assert left and right and len(left) == len(right)
    assert len({f["shape"] for f in left}) == len(left)
    assert Counter(f["shape"] for f in left) == Counter(f["shape"] for f in right)

    for frame in left:
        r0, c0, r1, c1 = frame["bbox"]
        interior = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        color = int(frame["color"])
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(grid[r, c]) == color:
                    interior[r - r0, c - c0] = True
        assert 1 <= int(interior.sum())
        frame["motif"] = interior

    return left, right, frame_cells, sep


def _gate_rows(grid: np.ndarray, sep: int, frame_cells: set[tuple[int, int]]) -> list[dict]:
    H, _W = grid.shape
    rows = []
    for r in range(H):
        if (r, sep - 1) in frame_cells or (r, sep + 1) in frame_cells:
            continue
        left_color = int(grid[r, sep - 1])
        right_color = int(grid[r, sep + 1])
        if left_color in {BG, SEP} or right_color in {BG, SEP}:
            continue
        rows.append({"row": int(r), "left_color": left_color, "right_color": right_color})
    return rows


def _frame_containing_row(frames: list[dict], row: int) -> dict:
    hits = [f for f in frames if f["bbox"][0] <= row <= f["bbox"][2]]
    assert len(hits) == 1, f"expected one frame for row {row}, got {len(hits)}"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer each left-frame motif into the same-sized right frame when the aligned separator-channel colors match, otherwise erase that right frame.

    Input structure:
    - The grid has a black background and one full-height grey vertical separator.
    - Left of the separator are 2 or 3 colored one-cell-thick hollow rectangles; each contains an interior motif in the same color.
    - Right of the separator are 2 or 3 colored hollow rectangles with matching outer sizes but empty interiors.
    - Each pair contributes one comparison row with two separator-adjacent reference pixels, one on each side of the separator.

    Rule:
    - Identify frame pairs by matching outer height and width.
    - For each comparison row, find the unique right frame spanning that row, then locate the same-sized left frame.
    - If the two reference-pixel colors on that row are equal, copy the left frame's interior motif into the right frame at the same relative positions, recolored to the right frame's border color.
    - If the two reference-pixel colors differ, erase the entire right frame to background.
    - Keep the left side, the separator, and the comparison pixels unchanged.

    Key insight:
    The separator-adjacent pixels are an independent gating channel: size decides which frames are paired, while color equality on the channel decides between motif transfer and full erasure.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    left_frames, right_frames, frame_cells, sep = _extract_frames(g)
    left_by_shape = {frame["shape"]: frame for frame in left_frames}
    gates = _gate_rows(g, sep, frame_cells)
    assert len(gates) == len(right_frames)

    seen_shapes = set()
    for gate in gates:
        right = _frame_containing_row(right_frames, gate["row"])
        shape = right["shape"]
        assert shape not in seen_shapes
        seen_shapes.add(shape)
        left = left_by_shape[shape]
        rr0, rc0, rr1, rc1 = right["bbox"]

        if gate["left_color"] != gate["right_color"]:
            out[rr0 : rr1 + 1, rc0 : rc1 + 1] = BG
            continue

        motif = left["motif"]
        right_color = int(right["color"])
        for dr in range(1, rr1 - rr0):
            for dc in range(1, rc1 - rc0):
                if bool(motif[dr, dc]):
                    out[rr0 + dr, rc0 + dc] = right_color

    assert len(seen_shapes) == len(right_frames)
    return out


def _random_motif(h: int, w: int) -> np.ndarray:
    safe_h = h - 4
    safe_w = w - 4
    area = safe_h * safe_w
    count = random.randint(3, min(6, area - 1))
    for _ in range(200):
        mask = np.zeros((h, w), dtype=bool)
        cells = random.sample(range(area), count)
        for idx in cells:
            rr, cc = divmod(idx, safe_w)
            mask[rr + 2, cc + 2] = True
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask
    raise RuntimeError("failed to build asymmetric motif")


def _paint_frame(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int, motif: np.ndarray | None = None) -> None:
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color
    if motif is not None:
        grid[top : top + h, left : left + w][motif] = color


def _check_no_overlap(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, (r0, c0, r1, c1) in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            s0, t0, s1, t1 = boxes[j]
            if not (r1 < s0 or s1 < r0 or c1 < t0 or t1 < c0):
                return False
    return True


def _oracle_output(inp: np.ndarray, left_specs: list[dict], right_specs: list[dict], gates: list[dict]) -> np.ndarray:
    out = inp.copy()
    left_by_shape = {spec["shape"]: spec for spec in left_specs}
    for gate in gates:
        right = next(spec for spec in right_specs if spec["gate_row"] == gate["row"])
        rr0, rc0 = right["top"], right["left"]
        h, w = right["shape"]
        if gate["left_color"] != gate["right_color"]:
            out[rr0 : rr0 + h, rc0 : rc0 + w] = BG
            continue
        motif = left_by_shape[right["shape"]]["motif"]
        out[rr0 : rr0 + h, rc0 : rc0 + w][_frame_mask(h, w)] = right["color"]
        out[rr0 + 1 : rr0 + h - 1, rc0 + 1 : rc0 + w - 1][motif[1:-1, 1:-1]] = right["color"]
    return out


def _ambiguous_row(left_specs: list[dict], row: int) -> int:
    return sum(spec["top"] <= row <= spec["top"] + spec["shape"][0] - 1 for spec in left_specs)


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.choices([2, 3], weights=[3, 2], k=1)[0]

    for _ in range(300):
        shapes = []
        seen = set()
        while len(shapes) < n:
            h = random.randint(6, 7)
            w = random.randint(6, 7)
            if (h, w) in seen:
                continue
            seen.add((h, w))
            shapes.append((h, w))

        a_shape, b_shape = shapes[0], shapes[1]
        max_right_w = max(w for _, w in shapes)
        left_a = 1
        left_b = left_a + a_shape[1] + 1
        left_c = left_a if n == 2 else left_b
        sep = max(left_a + a_shape[1] - 1, left_b + b_shape[1] - 1, left_c + (shapes[-1][1] if n == 3 else 0) - 1) + 3
        right_col = sep + 3
        W = right_col + max_right_w + 2
        if W > 30:
            continue

        H = random.randint(max(18, 6 * n), 26)
        if H > 30:
            continue

        amb_row = random.randint(6, H - 7)
        top_a = amb_row - random.randint(1, a_shape[0] - 2)
        top_b = amb_row - random.randint(1, b_shape[0] - 2)
        if top_a < 1 or top_b < 1 or top_a + a_shape[0] > H - 1 or top_b + b_shape[0] > H - 1:
            continue

        left_specs = [
            {"shape": a_shape, "top": top_a, "left": left_a},
            {"shape": b_shape, "top": top_b, "left": left_b},
        ]
        if n == 3:
            c_shape = shapes[2]
            c_top_choices = [r for r in range(1, H - c_shape[0]) if r + c_shape[0] - 1 < min(top_a, top_b) - 1 or r > max(top_a + a_shape[0] - 1, top_b + b_shape[0] - 1) + 1]
            if not c_top_choices:
                continue
            left_specs.append({"shape": c_shape, "top": random.choice(c_top_choices), "left": left_c})

        left_boxes = [
            (spec["top"], spec["left"], spec["top"] + spec["shape"][0] - 1, spec["left"] + spec["shape"][1] - 1)
            for spec in left_specs
        ]
        if not _check_no_overlap(left_boxes):
            continue
        if _ambiguous_row(left_specs, amb_row) < 2:
            continue

        center_shape = random.choice([a_shape, b_shape])
        other_shapes = [shape for shape in shapes if shape != center_shape]
        right_order = [other_shapes[0], center_shape]
        if n == 3:
            right_order = [other_shapes[0], center_shape, other_shapes[1]]

        center_h, _center_w = center_shape
        center_top = amb_row - random.randint(1, center_h - 2)
        if center_top < 1 or center_top + center_h > H - 1:
            continue

        right_specs = []
        if n == 2:
            top_shape = right_order[0]
            top_h, _ = top_shape
            top_choices = [r for r in range(1, center_top - top_h - 1)]
            bottom_choices = [r for r in range(center_top + center_h + 1, H - top_h)]
            if not top_choices and not bottom_choices:
                continue
            if top_choices and bottom_choices:
                top_first = random.choice([True, False])
            else:
                top_first = bool(top_choices)
            if top_first:
                placements = [(top_shape, random.choice(top_choices), None), (center_shape, center_top, amb_row)]
            else:
                placements = [(center_shape, center_top, amb_row), (top_shape, random.choice(bottom_choices), None)]
        else:
            top_shape, bottom_shape = right_order[0], right_order[2]
            top_h, _ = top_shape
            bottom_h, _ = bottom_shape
            top_choices = [r for r in range(1, center_top - top_h - 1)]
            bottom_choices = [r for r in range(center_top + center_h + 1, H - bottom_h)]
            if not top_choices or not bottom_choices:
                continue
            placements = [
                (top_shape, random.choice(top_choices), None),
                (center_shape, center_top, amb_row),
                (bottom_shape, random.choice(bottom_choices), None),
            ]

        used_rows = set()
        for shape, top, fixed_gate in placements:
            h, w = shape
            if fixed_gate is None:
                choices = [r for r in range(top + 1, top + h - 1) if r not in used_rows and r != amb_row]
                if not choices:
                    break
                gate_row = random.choice(choices)
            else:
                gate_row = fixed_gate
            used_rows.add(gate_row)
            right_specs.append(
                {
                    "shape": shape,
                    "top": top,
                    "left": right_col,
                    "color": None,
                    "gate_row": gate_row,
                }
            )
        else:
            right_boxes = [
                (spec["top"], spec["left"], spec["top"] + spec["shape"][0] - 1, spec["left"] + spec["shape"][1] - 1)
                for spec in right_specs
            ]
            if not _check_no_overlap(right_boxes):
                continue
            if any(not (spec["top"] <= spec["gate_row"] <= spec["top"] + spec["shape"][0] - 1) for spec in right_specs):
                continue

            colors = random.sample(FRAME_COLORS, 2 * n)
            left_colors = colors[:n]
            right_colors = colors[n:]
            for spec, color in zip(left_specs, left_colors):
                spec["color"] = color
                spec["motif"] = _random_motif(*spec["shape"])
            for spec, color in zip(right_specs, right_colors):
                spec["color"] = color

            gate_palette = FRAME_COLORS[:]
            matches = [True] * n
            mismatch_count = random.randint(1, n - 1)
            for idx in random.sample(range(n), mismatch_count):
                matches[idx] = False
            random.shuffle(matches)

            gates = []
            for spec, is_match in zip(right_specs, matches):
                left_gate = random.choice(gate_palette)
                if is_match:
                    right_gate = left_gate
                else:
                    right_gate = random.choice([c for c in gate_palette if c != left_gate])
                gates.append({"row": spec["gate_row"], "left_color": left_gate, "right_color": right_gate})

            inp = np.full((H, W), BG, dtype=int)
            inp[:, sep] = SEP
            for spec in left_specs:
                h, w = spec["shape"]
                _paint_frame(inp, spec["top"], spec["left"], h, w, spec["color"], spec["motif"])
            for spec in right_specs:
                h, w = spec["shape"]
                _paint_frame(inp, spec["top"], spec["left"], h, w, spec["color"], None)
            for gate in gates:
                inp[gate["row"], sep - 1] = gate["left_color"]
                inp[gate["row"], sep + 1] = gate["right_color"]

            expected = _oracle_output(inp, left_specs, right_specs, gates)

            if sum(g["left_color"] == g["right_color"] for g in gates) == 0:
                continue
            if sum(g["left_color"] != g["right_color"] for g in gates) == 0:
                continue
            if not any(_ambiguous_row(left_specs, gate["row"]) >= 2 for gate in gates):
                continue
            if not np.array_equal(main(inp), expected):
                continue
            return inp, expected

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        attempt = _generate_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate valid input")
