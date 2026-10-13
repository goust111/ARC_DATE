from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _find_monochrome_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append({"color": color, "coords": coords})

    return components


def _classify_input_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    shelves: list[dict] = []
    particles: list[dict] = []

    for component in _find_monochrome_components(grid):
        coords = component["coords"]
        color = int(component["color"])
        if len(coords) == 1:
            r, c = coords[0]
            particles.append({"row": int(r), "col": int(c), "color": color})
            continue

        rows = {r for r, _c in coords}
        cols = sorted(c for _r, c in coords)
        assert len(rows) == 1, "shelves must be one-row horizontal segments"
        left = cols[0]
        right = cols[-1]
        assert right - left + 1 == len(cols), "shelf cells must be contiguous"
        shelves.append(
            {
                "row": int(next(iter(rows))),
                "left": int(left),
                "right": int(right),
                "color": color,
            }
        )

    shelves.sort(key=lambda shelf: (shelf["row"], shelf["left"]))
    particles.sort(key=lambda particle: (particle["row"], particle["col"]))
    return shelves, particles


def _build_shelf_grid(shape: tuple[int, int], shelves: list[dict]) -> np.ndarray:
    shelf_grid = np.zeros(shape, dtype=int)
    for shelf in shelves:
        shelf_grid[shelf["row"], shelf["left"] : shelf["right"] + 1] = int(shelf["color"])
    return shelf_grid


def _trace_particle(
    shelf_grid: np.ndarray,
    start_r: int,
    start_c: int,
    color: int,
    *,
    shelf_ids: np.ndarray | None = None,
    mutate: bool,
    record_path: bool,
) -> dict:
    h, w = shelf_grid.shape
    cur_r = int(start_r)
    cur_c = int(start_c)
    path_cells: set[tuple[int, int]] = {(cur_r, cur_c)} if record_path else set()
    contact_shelves: set[int] = set()
    punch_cells: set[tuple[int, int]] = set()
    deflections = 0
    punches = 0

    while cur_r < h - 1:
        hit_row = None
        for rr in range(cur_r + 1, h):
            if int(shelf_grid[rr, cur_c]) != BACKGROUND:
                hit_row = int(rr)
                break

        if hit_row is None:
            if record_path:
                for rr in range(cur_r + 1, h):
                    path_cells.add((int(rr), cur_c))
            cur_r = h - 1
            break

        if record_path:
            for rr in range(cur_r + 1, hit_row + 1):
                path_cells.add((int(rr), cur_c))

        shelf_color = int(shelf_grid[hit_row, cur_c])
        if shelf_ids is not None:
            contact_shelves.add(int(shelf_ids[hit_row, cur_c]))

        if shelf_color == int(color):
            punches += 1
            punch_cells.add((hit_row, cur_c))
            if mutate:
                shelf_grid[hit_row, cur_c] = BACKGROUND
            cur_r = hit_row
            continue

        deflections += 1
        left = cur_c
        while left - 1 >= 0 and int(shelf_grid[hit_row, left - 1]) == shelf_color:
            left -= 1
        right = cur_c
        while right + 1 < w and int(shelf_grid[hit_row, right + 1]) == shelf_color:
            right += 1

        dist_left = cur_c - left
        dist_right = right - cur_c
        assert dist_left != dist_right, "hit point must be closer to one end"
        if dist_left < dist_right:
            exit_c = left - 1
            step = -1
        else:
            exit_c = right + 1
            step = 1
        assert 0 <= exit_c < w, "deflected particle must remain inside the grid"

        if record_path:
            for cc in range(cur_c + step, exit_c + step, step):
                path_cells.add((hit_row, int(cc)))

        cur_r = hit_row
        cur_c = int(exit_c)

    return {
        "final": (int(cur_r), int(cur_c)),
        "path_cells": path_cells,
        "contact_shelves": contact_shelves,
        "punch_cells": punch_cells,
        "deflections": int(deflections),
        "punches": int(punches),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each top pixel to the bottom while same-color shelves are pierced and different-color shelves deflect it sideways.

    Input structure:
    - A black background contains several monochrome horizontal one-row shelf segments.
    - A handful of colored single pixels are scattered in the upper part of the grid.
    - Shelf segments are separated from each other, and every shelf hit is strictly closer to one end than the other.

    Rule:
    - Each single pixel falls vertically.
    - When it reaches a shelf cell of the same color, it punches through that cell, erases the shelf cell to black, and keeps falling in the same column.
    - When it reaches a shelf cell of a different color, it slides along that shelf to the nearer end, steps past the end, and then resumes vertical falling; every pixel eventually stops on the bottom row.

    Key insight:
    - Shelves are not resting places: matching colors create holes, mismatching colors redirect the particle, and the final output is the shelves with any punched holes plus the particles relocated to the bottom row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    shelves, particles = _classify_input_objects(g)
    shelf_grid = _build_shelf_grid(g.shape, shelves)
    out = shelf_grid.copy()

    for particle in particles:
        trace = _trace_particle(
            out,
            particle["row"],
            particle["col"],
            particle["color"],
            mutate=True,
            record_path=False,
        )
        final_r, final_c = trace["final"]
        assert final_r == g.shape[0] - 1, "particles must settle on the bottom row"
        assert int(out[final_r, final_c]) == BACKGROUND, "final particle positions must be distinct"
        out[final_r, final_c] = int(particle["color"])

    return out


def _random_shelves(height: int, width: int, palette: tuple[int, ...]) -> list[dict] | None:
    max_count = min(4, max(3, (height - 8) // 3))
    count = random.randint(3, max_count)
    shelf_rows = sorted(random.sample(range(4, height - 2), count))
    if any(b - a < 3 for a, b in zip(shelf_rows, shelf_rows[1:])):
        return None

    lengths = [length for length in range(4, min(9, width - 2)) if length % 2 == 0]
    if not lengths:
        return None

    shelves: list[dict] = []
    for shelf_id, row in enumerate(shelf_rows, start=1):
        length = random.choice(lengths)
        left = random.randint(1, width - length - 2)
        color = random.choice(palette)
        shelves.append(
            {
                "id": int(shelf_id),
                "row": int(row),
                "left": int(left),
                "right": int(left + length - 1),
                "color": int(color),
            }
        )
    return shelves


def _build_generator_grids(shape: tuple[int, int], shelves: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    shelf_grid = np.zeros(shape, dtype=int)
    shelf_ids = np.zeros(shape, dtype=int)
    for shelf in shelves:
        row = int(shelf["row"])
        left = int(shelf["left"])
        right = int(shelf["right"])
        shelf_grid[row, left : right + 1] = int(shelf["color"])
        shelf_ids[row, left : right + 1] = int(shelf["id"])
    return shelf_grid, shelf_ids


def _enumerate_candidates(
    shelf_grid: np.ndarray,
    shelf_ids: np.ndarray,
    palette: tuple[int, ...],
) -> list[dict]:
    top_shelf_row = int(np.min(np.argwhere(shelf_ids > 0)[:, 0]))
    top_rows = list(range(min(2, top_shelf_row - 1)))
    candidates: list[dict] = []

    for start_r in top_rows:
        for start_c in range(shelf_grid.shape[1]):
            for color in palette:
                trace = _trace_particle(
                    shelf_grid.copy(),
                    start_r,
                    start_c,
                    int(color),
                    shelf_ids=shelf_ids,
                    mutate=True,
                    record_path=True,
                )
                interactions = trace["deflections"] + trace["punches"]
                if interactions == 0 or interactions > 2 or trace["deflections"] > 2:
                    continue

                kind = "deflect" if trace["deflections"] else "same_only"
                candidates.append(
                    {
                        "start": (int(start_r), int(start_c)),
                        "color": int(color),
                        "final": trace["final"],
                        "path_cells": trace["path_cells"],
                        "contact_shelves": trace["contact_shelves"],
                        "kind": kind,
                    }
                )

    random.shuffle(candidates)
    return candidates


def _compatible(candidate: dict, chosen: list[dict], used_cells: set[tuple[int, int]], used_shelves: set[int]) -> bool:
    if candidate["path_cells"] & used_cells:
        return False
    if candidate["contact_shelves"] & used_shelves:
        return False

    final_r, final_c = candidate["final"]
    for other in chosen:
        start_r, start_c = other["start"]
        if abs(candidate["start"][0] - start_r) <= 1 and abs(candidate["start"][1] - start_c) <= 1:
            return False
        _other_final_r, other_final_c = other["final"]
        if abs(final_c - other_final_c) <= 1:
            return False
    return final_r >= 0


def _select_particles(candidates: list[dict], total: int) -> list[dict] | None:
    def search(
        index: int,
        chosen: list[dict],
        used_cells: set[tuple[int, int]],
        used_shelves: set[int],
        seen_deflect: bool,
        seen_same_only: bool,
    ) -> list[dict] | None:
        if len(chosen) == total:
            if seen_deflect and seen_same_only:
                return list(chosen)
            return None

        remaining_slots = total - len(chosen)
        remaining_candidates = len(candidates) - index
        if remaining_candidates < remaining_slots:
            return None

        for i in range(index, len(candidates)):
            candidate = candidates[i]
            if not _compatible(candidate, chosen, used_cells, used_shelves):
                continue

            result = search(
                i + 1,
                chosen + [candidate],
                used_cells | candidate["path_cells"],
                used_shelves | candidate["contact_shelves"],
                seen_deflect or candidate["kind"] == "deflect",
                seen_same_only or candidate["kind"] == "same_only",
            )
            if result is not None:
                return result

        return None

    return search(0, [], set(), set(), False, False)


def _render_input(shape: tuple[int, int], shelves: list[dict], particles: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for shelf in shelves:
        grid[shelf["row"], shelf["left"] : shelf["right"] + 1] = int(shelf["color"])
    for particle in particles:
        r, c = particle["start"]
        grid[r, c] = int(particle["color"])
    return grid


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(16, 22)
    width = random.randint(14, 22)
    palette = tuple(random.sample(PALETTE, random.randint(3, 5)))

    shelves = _random_shelves(height, width, palette)
    if shelves is None:
        return None

    shelf_grid, shelf_ids = _build_generator_grids((height, width), shelves)
    candidates = _enumerate_candidates(shelf_grid, shelf_ids, palette)
    if not candidates:
        return None

    totals = [3, 4]
    random.shuffle(totals)
    particles = None
    for total in totals:
        particles = _select_particles(candidates, total)
        if particles is not None:
            break
    if particles is None:
        return None

    grid = _render_input((height, width), shelves, particles)
    output = main(grid)
    if np.array_equal(grid, output):
        return None
    if int(np.count_nonzero(output[-1] != BACKGROUND)) != len(particles):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to construct a valid shelf-and-particle grid")
