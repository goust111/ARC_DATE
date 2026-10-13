from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BACKGROUND = 0
GENERATION_RETRIES = 500
INNER_WIDTH_CHOICES = (4, 5, 6, 7)
BLOCK_HEIGHT_CHOICES = (1, 2, 3)
COUNT_PATTERNS = ((2, 1, 1), (2, 2, 1), (3, 1, 1))
GAP_CHOICES = (1, 2)
PALETTE = tuple(range(1, 10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _solid_block(grid: np.ndarray, obj: dict) -> dict | None:
    r0, c0, r1, c1 = obj["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    color = int(obj["color"])
    if int(obj["area"]) != h * w:
        return None
    if not np.all(sub == color):
        return None
    return {
        "kind": "block",
        "color": color,
        "bbox": obj["bbox"],
        "h": int(h),
        "w": int(w),
    }


def _open_container(grid: np.ndarray, bg: int, obj: dict) -> dict | None:
    r0, c0, r1, c1 = obj["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = sub.shape
    color = int(obj["color"])
    if h < 2 or w < 3:
        return None
    if int(obj["area"]) != 2 * h + w - 2:
        return None
    if not np.all(sub[:, 0] == color) or not np.all(sub[:, -1] == color):
        return None
    if not np.all(sub[-1, :] == color):
        return None
    if not np.all(sub[:-1, 1:-1] == bg):
        return None
    return {
        "kind": "container",
        "color": color,
        "bbox": obj["bbox"],
        "inner_width": int(w - 2),
        "inner_height": int(h - 1),
        "inner_left": int(c0 + 1),
        "inner_right": int(c1 - 1),
        "inner_top": int(r0),
        "floor_row": int(r1 - 1),
    }


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    containers: list[dict] = []
    blocks: list[dict] = []
    for obj in find_components(g, bg):
        container = _open_container(g, bg, obj)
        if container is not None:
            containers.append(container)
            continue
        block = _solid_block(g, obj)
        if block is not None:
            blocks.append(block)
            continue
        raise AssertionError("unexpected foreground component")
    assert len(containers) >= 2, "expected multiple containers"
    assert len(blocks) >= 3, "expected multiple blocks"
    widths = [container["inner_width"] for container in containers]
    assert len(widths) == len(set(widths)), "container inner widths must be unique"
    return bg, containers, blocks


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each solid rectangle into the open container whose inner width matches it, rotating when needed.

    Input structure:
    - Several one-cell-thick hollow containers sit in the lower part of the grid with their tops open upward.
    - Several floating monochrome solid rectangular blocks appear above them, separated from the containers and from each other.
    - Container inner widths are all distinct, and each block matches exactly one container by its width or by its height after a 90-degree rotation.

    Rule:
    - Keep the containers and background unchanged.
    - For each block, first match its current width to a container inner width; if that fails, rotate the block 90 degrees and match the rotated width instead.
    - Clear the block from its original position, then place it flush against the chosen container's inner walls and let it fall to the lowest available position.
    - If several blocks go into the same container, the block that was lower in the input lands first at the bottom, and higher ones stack above it.

    Key insight:
    - Horizontal position and color are decoys; only the metric relation between a block side length and a container inner width determines the destination, with rotation as the hidden extra freedom.
    """
    g = np.asarray(grid, dtype=int)
    bg, containers, blocks = _parse_scene(g)
    width_to_container = {container["inner_width"]: container for container in containers}

    out = g.copy()
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    assignments: dict[int, list[dict]] = defaultdict(list)
    for block in blocks:
        h = int(block["h"])
        w = int(block["w"])
        fits_now = w in width_to_container
        fits_rotated = h in width_to_container
        if fits_now and not fits_rotated:
            target_width = w
            placed_h, placed_w = h, w
        elif fits_rotated and not fits_now:
            target_width = h
            placed_h, placed_w = w, h
        elif fits_now and fits_rotated and h == w:
            target_width = w
            placed_h, placed_w = h, w
        else:
            raise AssertionError("block does not match exactly one container width")

        container = width_to_container[target_width]
        assert placed_w == container["inner_width"]
        assignments[target_width].append(
            {
                "color": int(block["color"]),
                "bbox": block["bbox"],
                "placed_h": int(placed_h),
                "placed_w": int(placed_w),
            }
        )

    for target_width, assigned in assignments.items():
        container = width_to_container[target_width]
        ordered = sorted(
            assigned,
            key=lambda item: (item["bbox"][2], item["bbox"][0], -item["bbox"][1]),
            reverse=True,
        )
        used_height = 0
        for item in ordered:
            top = container["floor_row"] - used_height - item["placed_h"] + 1
            left = container["inner_left"]
            assert top >= container["inner_top"], "container too shallow for assigned blocks"
            out[top : top + item["placed_h"], left : left + item["placed_w"]] = item["color"]
            used_height += item["placed_h"]
        assert used_height <= container["inner_height"]

    return out


def _render_container(grid: np.ndarray, container: dict) -> None:
    r0, c0, r1, c1 = container["bbox"]
    color = int(container["color"])
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r1, c0 : c1 + 1] = color


def _horizontal_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    left = max(a[1], b[1])
    right = min(a[3], b[3])
    return max(0, right - left + 1)


def _render_expected(inp: np.ndarray, bg: int, containers: list[dict], blocks: list[dict]) -> np.ndarray:
    out = inp.copy()
    width_to_container = {container["inner_width"]: container for container in containers}
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    grouped: dict[int, list[dict]] = defaultdict(list)
    for block in blocks:
        width_match = block["w"] if block["w"] in width_to_container else None
        height_match = block["h"] if block["h"] in width_to_container else None
        if width_match is not None:
            target_width = width_match
            placed_h = block["h"]
        else:
            assert height_match is not None
            target_width = height_match
            placed_h = block["w"]
        grouped[target_width].append(
            {
                "color": block["color"],
                "bbox": block["bbox"],
                "placed_h": int(placed_h),
            }
        )

    for width, items in grouped.items():
        container = width_to_container[width]
        items = sorted(items, key=lambda item: item["bbox"][2], reverse=True)
        used_height = 0
        for item in items:
            top = container["floor_row"] - used_height - item["placed_h"] + 1
            out[top : top + item["placed_h"], container["inner_left"] : container["inner_right"] + 1] = int(item["color"])
            used_height += item["placed_h"]
    return out


def _choose_blocks(rng: random.Random, containers: list[dict]) -> list[dict]:
    counts = list(rng.choice(COUNT_PATTERNS))
    rng.shuffle(counts)
    total = sum(counts)

    rotated_flags = [bool(rng.randint(0, 1)) for _ in range(total)]
    if not any(rotated_flags):
        rotated_flags[rng.randrange(total)] = True
    if all(rotated_flags):
        rotated_flags[rng.randrange(total)] = False

    blocks: list[dict] = []
    flag_idx = 0
    for container, count in zip(containers, counts):
        heights = [rng.choice(BLOCK_HEIGHT_CHOICES) for _ in range(count)]
        container["inner_height"] = max(4, sum(heights) + rng.randint(1, 3))
        r0, c0, _, _ = container["bbox"]
        container["bbox"] = (r0, c0, r0 + container["inner_height"], c0 + container["inner_width"] + 1)
        container["floor_row"] = container["bbox"][2] - 1
        container["inner_top"] = r0
        for final_h in heights:
            rotated = rotated_flags[flag_idx]
            flag_idx += 1
            if rotated:
                input_h = container["inner_width"]
                input_w = final_h
            else:
                input_h = final_h
                input_w = container["inner_width"]
            wrong_colors = [other["color"] for other in containers if other["color"] != container["color"]]
            blocks.append(
                {
                    "target_width": int(container["inner_width"]),
                    "target_color": int(container["color"]),
                    "color": int(rng.choice(wrong_colors)),
                    "rotated": bool(rotated),
                    "final_h": int(final_h),
                    "h": int(input_h),
                    "w": int(input_w),
                }
            )
    rng.shuffle(blocks)
    return blocks


def _place_block_rectangles(
    rng: random.Random,
    h: int,
    w: int,
    blocks: list[dict],
    top_limit: int,
) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    for block in blocks:
        bh = int(block["h"])
        bw = int(block["w"])
        placed = False
        for _ in range(200):
            r0 = rng.randint(1, top_limit - bh + 1)
            c0 = rng.randint(1, w - bw - 1)
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + bh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + bw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + bh, c0 : c0 + bw] = True
            block["bbox"] = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            placed = True
            break
        if not placed:
            return False
    return True


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    widths = list(rng.sample(INNER_WIDTH_CHOICES, 3))
    rng.shuffle(widths)
    colors = rng.sample(PALETTE, 3)
    gaps = [rng.choice(GAP_CHOICES) for _ in range(2)]

    containers = []
    x = 1
    for width, color in zip(widths, colors):
        outer_w = width + 2
        containers.append(
            {
                "inner_width": int(width),
                "inner_height": 0,
                "color": int(color),
                "bbox": (0, x, 0, x + outer_w - 1),
                "inner_left": int(x + 1),
                "inner_right": int(x + width),
                "inner_top": 0,
                "floor_row": 0,
            }
        )
        x += outer_w
        if len(containers) < 3:
            x += gaps[len(containers) - 1]
    width_total = x + 1
    if width_total > 30:
        return None

    blocks = _choose_blocks(rng, containers)
    outer_heights = [container["inner_height"] + 1 for container in containers]
    max_outer_h = max(outer_heights)
    tallest_block = max(block["h"] for block in blocks)
    height = max(18, max_outer_h + tallest_block + len(blocks) + 5)
    if height > 30:
        return None

    base_bottom = height - 2
    for container in containers:
        outer_h = container["inner_height"] + 1
        r0 = base_bottom - outer_h + 1
        c0 = container["bbox"][1]
        container["bbox"] = (r0, c0, base_bottom, c0 + container["inner_width"] + 1)
        container["inner_left"] = c0 + 1
        container["inner_right"] = c0 + container["inner_width"]
        container["inner_top"] = r0
        container["floor_row"] = base_bottom - 1

    top_limit = min(container["bbox"][0] for container in containers) - 2
    if top_limit < tallest_block + 1:
        return None
    if not _place_block_rectangles(rng, height, width_total, blocks, top_limit):
        return None

    by_target: dict[int, list[dict]] = defaultdict(list)
    for block in blocks:
        by_target[block["target_width"]].append(block)

    if not any(len(items) >= 2 for items in by_target.values()):
        return None
    if not any(block["rotated"] for block in blocks):
        return None
    if not any(not block["rotated"] for block in blocks):
        return None
    if not all(len({item["bbox"][2] for item in items}) == len(items) for items in by_target.values()):
        return None

    width_to_container = {container["inner_width"]: container for container in containers}
    misaligned = False
    for block in blocks:
        container = width_to_container[block["target_width"]]
        if _horizontal_overlap(block["bbox"], container["bbox"]) == 0:
            misaligned = True
        if block["color"] == container["color"]:
            return None
        if block["w"] == block["h"]:
            return None
    if not misaligned:
        return None

    inp = np.full((height, width_total), BACKGROUND, dtype=int)
    for container in containers:
        _render_container(inp, container)
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = int(block["color"])

    bg, parsed_containers, parsed_blocks = _parse_scene(inp)
    if len(parsed_containers) != len(containers) or len(parsed_blocks) != len(blocks):
        return None

    expected = _render_expected(inp, bg, parsed_containers, parsed_blocks)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
