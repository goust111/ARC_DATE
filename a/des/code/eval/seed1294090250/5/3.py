import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
SOURCE_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _detect_nodes(g: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    h, w = g.shape
    nodes: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    cell_to_node: dict[tuple[int, int], int] = {}
    for r in range(h - 1):
        for c in range(w - 1):
            cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            if any(cell in occupied for cell in cells):
                continue
            vals = [int(g[cell]) for cell in cells]
            if vals.count(ORANGE) == 1 and vals.count(GREY) == 3:
                orange_cell = cells[vals.index(ORANGE)]
                idx = len(nodes)
                node = {
                    "r0": r,
                    "c0": c,
                    "cells": tuple(cells),
                    "orange": orange_cell,
                    "orange_local": (orange_cell[0] - r, orange_cell[1] - c),
                }
                nodes.append(node)
                for cell in cells:
                    occupied.add(cell)
                    cell_to_node[cell] = idx
    return nodes, cell_to_node


def _source_cells(g: np.ndarray) -> list[tuple[int, int, int]]:
    sources = []
    for r, c in np.argwhere((g != BG) & (g != GREY) & (g != ORANGE)):
        sources.append((int(r), int(c), int(g[r, c])))
    return sources


def _initial_step(g: np.ndarray, r: int, c: int) -> tuple[int, int, int, int] | None:
    h, w = g.shape
    options = []
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == GREY:
            options.append((rr, cc, dr, dc))
    if len(options) != 1:
        return None
    return options[0]


def _node_exit(node: dict, dr: int, dc: int) -> tuple[int, int, int, int]:
    r0 = int(node["r0"])
    c0 = int(node["c0"])
    orange_r, orange_c = node["orange_local"]
    if dc != 0:
        out_dr = -1 if orange_r == 0 else 1
        out_dc = 0
        rr = r0 - 1 if out_dr < 0 else r0 + 2
        cc = c0 + orange_c
    else:
        out_dr = 0
        out_dc = -1 if orange_c == 0 else 1
        rr = r0 + orange_r
        cc = c0 - 1 if out_dc < 0 else c0 + 2
    return rr, cc, out_dr, out_dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color through grey circuit lines. A 2x2 node has one
    orange turn marker: its row selects up/down for horizontal entry, and its
    column selects left/right for vertical entry. Grey node cells are colored by
    the signal; orange markers remain fixed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    nodes, cell_to_node = _detect_nodes(g)
    out = g.copy()

    for sr, sc, color in _source_cells(g):
        step = _initial_step(g, sr, sc)
        if step is None:
            continue
        r, c, dr, dc = step
        for _ in range(h * w * 4):
            if not (0 <= r < h and 0 <= c < w):
                break
            if (r, c) in cell_to_node:
                node = nodes[cell_to_node[(r, c)]]
                for rr, cc in node["cells"]:
                    if int(g[rr, cc]) == GREY:
                        if int(out[rr, cc]) not in (GREY, color):
                            break
                        out[rr, cc] = color
                r, c, dr, dc = _node_exit(node, dr, dc)
                continue
            if int(g[r, c]) != GREY:
                break
            if int(out[r, c]) not in (GREY, color):
                break
            out[r, c] = color
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if (nr, nc) not in cell_to_node and int(g[nr, nc]) != GREY:
                break
            r, c = nr, nc
    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(12, 18)
        w = random.randint(14, 20)
        node_r = random.randint(4, h - 6)
        node_c = random.randint(5, w - 5)
        orange_r = random.randint(0, 1)
        orange_c = random.randint(0, 1)
        entry_r = node_r + (1 - orange_r)
        exit_c = node_c + orange_c

        g = np.full((h, w), BG, dtype=int)
        g[entry_r, 1:node_c] = GREY
        g[node_r : node_r + 2, node_c : node_c + 2] = GREY
        g[node_r + orange_r, node_c + orange_c] = ORANGE

        if orange_r == 0:
            g[1:node_r, exit_c] = GREY
        else:
            g[node_r + 2 : h - 1, exit_c] = GREY

        g[entry_r, 0] = random.choice(SOURCE_COLORS)
        return g
    raise RuntimeError("failed to generate seed1294090250/5/3 input")
