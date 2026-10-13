import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def area(self) -> int:
        return self.height * self.width


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_from_component(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> Frame | None:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(cells) != expected:
        return None
    cell_set = set(cells)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border:
                if int(grid[r, c]) != color or (r, c) not in cell_set:
                    return None
            elif (r, c) in cell_set:
                return None
    return Frame(color=color, r0=r0, c0=c0, r1=r1, c1=c1)


def _find_frames(grid: np.ndarray, bg: int) -> list[Frame]:
    frames: list[Frame] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            frame = _frame_from_component(grid, comp, color)
            if frame is not None:
                frames.append(frame)
    return frames


def _contains(outer: Frame, inner: Frame) -> bool:
    return (
        outer.r0 + 1 < inner.r0
        and outer.c0 + 1 < inner.c0
        and inner.r1 < outer.r1 - 1
        and inner.c1 < outer.c1 - 1
    )


def _parent_indices(frames: list[Frame]) -> list[int]:
    parents = [-1] * len(frames)
    for i, inner in enumerate(frames):
        best_parent = -1
        best_area = None
        for j, outer in enumerate(frames):
            if i == j or not _contains(outer, inner):
                continue
            if best_area is None or outer.area < best_area:
                best_area = outer.area
                best_parent = j
        parents[i] = best_parent
    return parents


def _leaf_contents(grid: np.ndarray, leaf: Frame, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    coords: list[tuple[int, int]] = []
    offsets: list[tuple[int, int]] = []
    for dr in range(leaf.height - 2):
        for dc in range(leaf.width - 2):
            r = leaf.r0 + 1 + dr
            c = leaf.c0 + 1 + dc
            if int(grid[r, c]) != bg:
                coords.append((r, c))
                offsets.append((dr, dc))
    return coords, offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the innermost container's contents outward through nested rectangular frames, recoloring them by the crossed border.

    Input structure:
    - The background is the dominant color.
    - Foreground objects are one-pixel-thick rectangular frame containers; some appear as strict nesting chains of depth two or three.
    - Only the innermost frame of a chain contains scattered content pixels, and those content pixels sit inside the frame interior rather than on borders.

    Rule:
    - Detect every rectangular frame and reconstruct parent-child containment by strict nesting.
    - For each leaf frame that has at least one parent, read the content pattern as offsets from the leaf interior's top-left anchor.
    - Remove those original content pixels, then place the same offset pattern into the outermost container of that chain using that container interior's top-left anchor; the migrated pixels take the outermost border color.
    - Keep every frame border unchanged.

    Key insight:
    - The important structure is the nesting hierarchy of rectangular borders: repeated outward migration preserves offsets but the surviving final color is determined by the outermost crossed container.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _find_frames(g, bg)
    if not frames:
        return g.copy()

    parents = _parent_indices(frames)
    children: list[list[int]] = [[] for _ in frames]
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx != -1:
            children[parent_idx].append(child_idx)

    out = g.copy()
    for leaf_idx, leaf in enumerate(frames):
        if children[leaf_idx] or parents[leaf_idx] == -1:
            continue

        content_coords, offsets = _leaf_contents(g, leaf, bg)
        if not offsets:
            continue

        chain = [leaf_idx]
        cur = parents[leaf_idx]
        while cur != -1:
            chain.append(cur)
            cur = parents[cur]

        root = frames[chain[-1]]
        blocker = frames[chain[-2]]

        for r, c in content_coords:
            out[r, c] = bg

        for dr, dc in offsets:
            rr = root.r0 + 1 + dr
            cc = root.c0 + 1 + dc
            assert root.r0 < rr < root.r1 and root.c0 < cc < root.c1
            assert not (blocker.r0 <= rr <= blocker.r1 and blocker.c0 <= cc <= blocker.c1)
            out[rr, cc] = root.color

    return out


MAX_GENERATE_ATTEMPTS = 400
GROUP_COUNT_CHOICES = (1, 2)
DEPTH_CHOICES = (2, 3)
INSET_RANGE = (2, 4)
LEAF_INTERIOR_BY_DEPTH = {
    2: (2, 5),
    3: (2, 4),
}


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _shift_boxes(boxes: list[tuple[int, int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes]


def _valid_offsets_for_chain(
    leaf_h: int,
    leaf_w: int,
    boxes_inner_to_outer: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    blocked: list[tuple[int, int, int, int]] = []
    for child_box, parent_box in zip(boxes_inner_to_outer, boxes_inner_to_outer[1:]):
        pr0, pc0, _, _ = parent_box
        cr0, cc0, cr1, cc1 = child_box
        blocked.append((cr0 - pr0 - 1, cc0 - pc0 - 1, cr1 - pr0 - 1, cc1 - pc0 - 1))

    valid: list[tuple[int, int]] = []
    for dr in range(leaf_h):
        for dc in range(leaf_w):
            if any(br0 <= dr <= br1 and bc0 <= dc <= bc1 for br0, bc0, br1, bc1 in blocked):
                continue
            valid.append((dr, dc))
    return valid


def _make_group_patch(
    rng: random.Random,
    bg: int,
    frame_colors: list[int],
    content_color: int,
    max_side: int,
) -> np.ndarray | None:
    depth = len(frame_colors)
    min_leaf, max_leaf = LEAF_INTERIOR_BY_DEPTH[depth]

    for _ in range(120):
        leaf_h = rng.randint(min_leaf, max_leaf)
        leaf_w = rng.randint(min_leaf, max_leaf)

        cur_h = leaf_h + 2
        cur_w = leaf_w + 2
        boxes: list[tuple[int, int, int, int]] = [(0, 0, cur_h - 1, cur_w - 1)]

        for _ in range(depth - 1):
            top = rng.randint(*INSET_RANGE)
            left = rng.randint(*INSET_RANGE)
            bottom = rng.randint(*INSET_RANGE)
            right = rng.randint(*INSET_RANGE)
            boxes = _shift_boxes(boxes, top, left)
            cur_h += top + bottom
            cur_w += left + right
            boxes.append((0, 0, cur_h - 1, cur_w - 1))

        if cur_h > max_side or cur_w > max_side:
            continue

        boxes_inner_to_outer = sorted(boxes, key=lambda box: (box[2] - box[0], box[3] - box[1]))
        valid_offsets = _valid_offsets_for_chain(leaf_h, leaf_w, boxes_inner_to_outer)
        if len(valid_offsets) < 2:
            continue

        max_count = min(6, len(valid_offsets))
        min_count = 2
        if max_count > 2:
            max_count = max_count - rng.randint(0, 1)
        count = rng.randint(min_count, max_count)
        offsets = rng.sample(valid_offsets, count)

        patch = np.full((cur_h, cur_w), bg, dtype=int)
        boxes_outer_to_inner = sorted(boxes, key=lambda box: (box[2] - box[0], box[3] - box[1]), reverse=True)
        for color, box in zip(frame_colors, boxes_outer_to_inner):
            _draw_frame(patch, box, color)

        leaf_box = boxes_inner_to_outer[0]
        for dr, dc in offsets:
            patch[leaf_box[0] + 1 + dr, leaf_box[1] + 1 + dc] = content_color

        return patch

    return None


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    grid[top : top + ph, left : left + pw] = patch


def _single_patch_canvas(rng: random.Random, bg: int, patch: np.ndarray) -> np.ndarray | None:
    ph, pw = patch.shape
    top = rng.randint(1, 5)
    bottom = rng.randint(1, 5)
    left = rng.randint(1, 5)
    right = rng.randint(1, 5)
    extra_h = rng.randint(0, 5)
    extra_w = rng.randint(0, 5)
    h = ph + top + bottom + extra_h
    w = pw + left + right + extra_w
    if h > 30 or w > 30:
        return None
    grid = np.full((h, w), bg, dtype=int)
    r = top + rng.randint(0, extra_h)
    c = left + rng.randint(0, extra_w)
    _paste_patch(grid, patch, r, c)
    return grid


def _two_patch_canvas(rng: random.Random, bg: int, patches: list[np.ndarray]) -> np.ndarray | None:
    patches = patches[:]
    rng.shuffle(patches)
    (ph1, pw1), (ph2, pw2) = patches[0].shape, patches[1].shape

    layouts: list[tuple[str, dict[str, int]]] = []

    gap = rng.randint(2, 5)
    top = rng.randint(1, 4)
    bottom = rng.randint(1, 4)
    left = rng.randint(1, 4)
    right = rng.randint(1, 4)
    extra_h = rng.randint(0, 4)
    extra_w = rng.randint(0, 4)
    h = max(ph1, ph2) + top + bottom + extra_h
    w = pw1 + pw2 + gap + left + right + extra_w
    if h <= 30 and w <= 30:
        layouts.append(
            (
                "horizontal",
                {
                    "gap": gap,
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                    "extra_h": extra_h,
                    "extra_w": extra_w,
                    "h": h,
                    "w": w,
                },
            )
        )

    gap = rng.randint(2, 5)
    top = rng.randint(1, 4)
    bottom = rng.randint(1, 4)
    left = rng.randint(1, 4)
    right = rng.randint(1, 4)
    extra_h = rng.randint(0, 4)
    extra_w = rng.randint(0, 4)
    h = ph1 + ph2 + gap + top + bottom + extra_h
    w = max(pw1, pw2) + left + right + extra_w
    if h <= 30 and w <= 30:
        layouts.append(
            (
                "vertical",
                {
                    "gap": gap,
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                    "extra_h": extra_h,
                    "extra_w": extra_w,
                    "h": h,
                    "w": w,
                },
            )
        )

    if not layouts:
        return None

    mode, spec = rng.choice(layouts)
    grid = np.full((spec["h"], spec["w"]), bg, dtype=int)

    if mode == "horizontal":
        x1 = spec["left"] + rng.randint(0, spec["extra_w"])
        x2 = x1 + pw1 + spec["gap"]
        y1 = spec["top"] + rng.randint(0, spec["h"] - spec["top"] - spec["bottom"] - ph1)
        y2 = spec["top"] + rng.randint(0, spec["h"] - spec["top"] - spec["bottom"] - ph2)
        _paste_patch(grid, patches[0], y1, x1)
        _paste_patch(grid, patches[1], y2, x2)
    else:
        y1 = spec["top"] + rng.randint(0, spec["extra_h"])
        y2 = y1 + ph1 + spec["gap"]
        x1 = spec["left"] + rng.randint(0, spec["w"] - spec["left"] - spec["right"] - pw1)
        x2 = spec["left"] + rng.randint(0, spec["w"] - spec["left"] - spec["right"] - pw2)
        _paste_patch(grid, patches[0], y1, x1)
        _paste_patch(grid, patches[1], y2, x2)

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    group_count = rng.choice(GROUP_COUNT_CHOICES)
    depths = [rng.choice(DEPTH_CHOICES) for _ in range(group_count)]
    needed_colors = 1 + sum(depth + 1 for depth in depths)
    if needed_colors > 10:
        return None

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette.pop()

    max_side = 13 if group_count == 2 else 22
    patches: list[np.ndarray] = []
    for depth in depths:
        frame_colors = [palette.pop() for _ in range(depth)]
        content_color = palette.pop()
        patch = _make_group_patch(rng, bg, frame_colors, content_color, max_side=max_side)
        if patch is None:
            return None
        patches.append(patch)

    if group_count == 1:
        return _single_patch_canvas(rng, bg, patches[0])
    return _two_patch_canvas(rng, bg, patches)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if isinstance(out, np.ndarray) and out.shape == grid.shape and np.any(out != grid):
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATE_ATTEMPTS} attempts")
