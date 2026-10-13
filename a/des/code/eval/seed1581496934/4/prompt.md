你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="eee78d87" lines="40-55">
<description>
Decode the colored 3x3 contour motif into a 16x16 periodic lattice with the motif stamped in maroon at the center.

Input structure:
- The grid is 6x6 with orange background and one non-orange color.
- The non-orange pixels have a tight 3x3 bbox containing one contour glyph.
- The glyph is one of the task's period-3 contour forms: plus, H, X, or four corners.

Rule:
- Decode the 3x3 glyph into the period-3 orange phase tile.
- Fill a black 16x16 canvas by repeating that orange phase tile.
- Tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon in the centered 6x6 region.

Key insight:
- The colored object is not copied at its input location; it is a small contour code whose phase tells where orange appears in the repeated lattice.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="fc7cae8d" lines="9-21">
<description>
Align a central rectangle to two colored frame-edge controls via rotation/reflection.

Input structure:
- bg grid with two solid-colored border edges (distinct non-bg colors).
- A central filled rectangle with 1px colored guide edges on two sides, colors matching the frame controls.

Rule:
- Rotate/mirror the object so its guide edges face the same sides as the matching frame controls.
- Crop off the 1px guide edges from aligned sides.

Key insight: Guide-edge colors encode target frame sides; the correct dihedral transform matches both simultaneously.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="136b0064" lines="25-39">
<description>
Draw a polyline by decoding 3x3 glyph instructions read across a divided grid.

Input structure:
- Left of a full-height vertical divider: 3x3 colored glyphs on bg.
- Right of divider: a single gray (5) start pixel.

Rule:
- Remove the divider; shift the gray start into the left canvas.
- Read glyphs in column-major order (left→right, top→bottom).
- Each glyph's shape (pixel area) encodes a segment: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
- Append segments sequentially, stepping down one cell between each.

Key insight: Glyph pixel count (area) uniquely determines both segment length and direction — the shape itself is the instruction.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="7491f3cf" lines="9-24">
<description>
Composite two pattern tiles into a fourth panel using flood-fill from an indicator pixel.

Input structure:
- Four 5×5 panels in one row, border-separated.
- Panel 0: cue pixels = wall boundary (8-connected, edge-to-edge, through center)
           + one 8-isolated indicator pixel.
- Panels 1–2: source patterns. Panel 3: blank (to fill).

Rule:
- Flood-fill (4-conn) from indicator, blocked by wall → inside region.
- Inside → non-bg from panel 1; outside → non-bg from panel 2.
- Wall → panel 1 if non-bg; panel 2 fallback when wall is 4-connected.

Key insight: The cue panel's isolated indicator and wall partition the tile into inside/outside masks that select between the two source panels.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="db0c5428" lines="32-51">
<description>
Expand a 3x3 block motif into a 5x5 block motif around its missing center.

Input structure:
- Teal background surrounds one 9x9 foreground motif.
- The motif is a 3x3 array of 3x3 blocks; the middle block is empty teal.
- Colored pixels immediately around the empty middle block encode the border
  of the missing center block.

Rule:
- Keep the original 9x9 motif in place.
- Copy each of the eight surrounding 3x3 blocks to the opposite outer slot
  of the implied 5x5 block grid, without rotating it.
- Fill the center block from the eight pixels around the hole; its center is
  the motif's majority foreground color.

Key insight: The visible 9x9 motif is the middle of a larger periodic
block-level pattern, and the empty center is reconstructed from its immediate
inner boundary.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="28a6681f" lines="8-23">
<description>
Simulate water pixels falling under gravity into basins formed by colored obstacles.

Input structure:
- bg = 0. Water pixels = color 1 (blue).
- Other colors form static solid structures (walls, steps, containers).
- Water starts in unstable positions above or beside basins.

Rule:
- Each water pixel falls straight down through bg cells.
- When blocked below, it slides diagonally through the unique open side cell.
- When resting on a surface, it slides horizontally toward the nearest gap where it can fall again.
- Process repeats until no water pixel moves.

Key insight: Water seeks the deepest reachable basin via diagonal and horizontal sliding, not just vertical fall.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="16b78196" lines="6-20">
<description>
Assemble colored sprites into a central bar's notches, chaining successive pieces along fixed-width strips.

Input structure:
- bg = most common color. Largest monochrome component is a solid "bar" with boundary notches (bg holes opening to its bbox edge).
- Several smaller colored sprites scattered around the bar.

Rule:
- Each bar notch is filled by a root piece whose shape exactly covers the hole.
- The root defines a strip (column-span or row-span) for its chain.
- Successive pieces fill the bg cells inside the previous piece's bbox, locked to the same strip width.
- Multiple chains grow independently from different bar notches.

Key insight: The root piece's extent perpendicular to the chain direction locks the strip width for all subsequent pieces in that chain.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="a395ee82" lines="9-24">
<description>
Stamp translated copies of a monochrome template guided by a singleton pixel layout.

Input structure:
- bg = background. A large monochrome shape (template, color T).
- Singleton pixels elsewhere: exactly one is color T (layout origin); the rest share color O (overlap color).
- Singletons sit on a uniform-spacing lattice.

Rule:
- Convert each singleton's offset from the origin into coarse lattice units.
- Place a template copy at each offset, scaled by template height/width.
- The origin's copy is painted O instead of T.
- Any pixel covered by two or more copies is also recolored to O.

Key insight: The single T-colored singleton identifies both the origin position and the "self" copy, which is uniquely recolored.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="58490d8a" lines="9-21">
<description>
Count colored objects and record tallies inside a legend box.

Input structure:
- A black rectangular legend box with key-color pixels in its leftmost non-black column.
- Outside the box, distinct objects of each key color are scattered across the grid.

Rule:
- For each key color, count separate same-colored objects outside the legend box.
- Output the legend box repainted: each key row gets one dot per object, spaced every other cell rightward from the key pixel.

Key insight: The legend is both the color index and the output canvas — object counts become dot sequences on the matching row.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="c4d067a0" lines="9-23">
<description>
Complete a grid of colored blocks to match a minimap legend in the top-left corner.

Input structure:
- bg = most common color.
- Top-left corner: a small sprite of single colored pixels forming a minimap pattern on a regular lattice.
- Main area: uniformly colored square blocks (NxN) arranged on a regular grid; some positions are empty (bg).

Rule:
- The legend pixels define a color pattern; each pixel's lattice position maps to a block position in the main grid via a fixed scale factor and offset.
- Existing colored blocks confirm the scale/offset alignment.
- For every legend pixel whose corresponding grid position is empty, place a new block of that color.

Key insight: The top-left pixel pattern is a scaled-down minimap of the block layout; the mapping is inferred from blocks already present.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="36a08778" lines="8-21">
<description>
Simulate water falling from top-edge sources, pooling on shelves, and cascading off edges.

Input structure:
- bg = most common color; water = rarest non-bg; shelf = other non-bg.
- Sources: water clusters touching top row. Shelves: horizontal segments.

Rule:
- Water falls down from sources, painting bg cells as water.
- On hitting a shelf, it spreads across the shelf span plus one cell past each end.
- From those overflow positions, water resumes falling independently.

Key insight: Each shelf splits a stream into two at its edges, creating a branching waterfall tree.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="271d71e2" lines="9-22">
<description>
Move bordered containers toward same-colored destination markers, charging gray cells to orange.

Input structure:
- bg = 6 (magenta). Black-bordered (0) rectangles with gray (5) / orange (7) interiors.
- A head sprite (9) sits adjacent to one container edge; a matching destination sprite lies along the movement axis.

Rule:
- Each container translates toward its destination along the head-to-dest axis.
- Steps = min(head-to-dest distance, number of gray interior cells).
- Gray cells become orange from the head-facing side inward, one per step.

Key insight: Gray cell count caps movement — a container stops short if it runs out of gray cells to charge.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="8f3a5a89" lines="22-36">
<description>
Paint an inner border around the region reachable from a corner marker pixel.

Input structure:
- bg = most common color; wall color (blue) forms barriers dividing the grid.
- A unique single pixel (magenta) sits on one corner — the start marker.
- Some wall-colored pixels are structural (border-touching, enclosing regions); others are decorations.

Rule:
- Flood-fill from the marker corner to find all reachable bg cells, blocked by structural walls.
- Erase wall-colored decorations hidden behind structural walls (not reachable from marker side).
- Paint orange (color 7) on every reachable bg cell adjacent to a wall or grid edge — forming a 1-cell inner border.

Key insight: Only border-connected wall components act as real barriers; interior wall fragments are decorations to be removed.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="cb2d8a2c" lines="13-26">
<description>
Extend a green path from one grid edge to the opposite, routing around red bars at a distance dictated by their blue pixel count.

Input structure:
- bg = azure (8). Red/blue bars (color 2/1) act as obstacles.
- A single green (3) pixel sits on a grid border — the path start.

Rule:
- Each bar's keep-out radius equals its number of blue pixels; the path must stay farther than that from every bar cell.
- The green pixel extends inward toward the opposite edge, turning at right angles to navigate around forbidden zones.
- After the path is drawn, all blue pixels become red.

Key insight: Blue pixels on a bar encode its danger radius — more blue means a wider no-go zone the path must respect.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="7b3084d4" lines="253-266">
<description>
Assemble colored rectangular sprites into a single tiled rectangle, anchored by a marker pixel.

Input structure:
- bg = most common color. Several solid-colored rectangular sprites scattered on the grid.
- One sprite has a unique single-pixel marker (different color) at one corner — this is the anchor.

Rule:
- Identify the anchor sprite (contains the marker pixel) and the remaining pieces.
- Rotate the anchor so the marker pixel lands at position (0,0) of the output.
- Pack all remaining pieces (allowing 90° rotations) to tile a complete rectangle with no gaps.

Key insight: The lone marker pixel picks the anchor and fixes the output orientation — it must sit at the top-left corner.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="a47bf94d" lines="178-189">
<description>
Transmit colored tokens along wire networks, converting each to its complementary 3×3 pattern.

Input structure:
- bg = background. Non-bg cells form wire networks and 3×3 tokens (solid blocks or X/diagonal patterns) near wire endpoints.

Rule:
- Every solid 3×3 token is replaced in-place by a diamond (center + 4 cardinal neighbors).
- Each token near a wire endpoint sends its complement to the opposite endpoint: solid emits X, X emits diamond.

Key insight: Token shape cycles solid → X → diamond; wires route each complement to the far end.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="45a5af55" lines="5-18">
<description>
Build nested square frames from a stack of horizontal color stripes.

Input structure:
- Full-width horizontal stripes of distinct non-bg colors, stacked vertically.
- Each stripe has uniform color and variable height (1-4 rows).

Rule:
- Read stripes bottom to top.
- Bottom stripe becomes the center solid square with side length = stripe height.
- Each subsequent stripe wraps the current shape with a square frame of thickness = stripe height, in that stripe's color.

Key insight: Stripe height doubles as both frame thickness and (for the innermost) center-square side length; output dimensions are fully determined by the stripe heights.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="89565ca0" lines="120-136">
<description>
Convert each colored partitioned frame into a horizontal bar whose length is its number of enclosed regions.

Input structure:
- Black background with several axis-aligned colored line drawings.
- The line drawings are damaged by scattered pixels of one non-background noise color and by overlaps with other drawings.
- Each real drawing is a rectangular partitioned frame enclosing one or more background regions.

Rule:
- For each non-background color, reconstruct its horizontal and vertical wall segments from dense row/column evidence.
- Count the 4-connected background regions fully enclosed by those reconstructed walls.
- The color with no enclosed region is the noise/pad color.
- Sort drawing colors by enclosed-region count; equal counts are ordered by which reconstructed frame covers the other.

Key insight:
- The scattered color is not an object; it only pads the histogram, while each true object's room count is stable after rectilinear wall reconstruction.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="38007db0" lines="9-22">
<description>
Extract the unique cell from each lattice row into a single-column lattice.

Input structure:
- Grid divided into equal square cells by uniform gridlines of color L.
- Cell interiors filled with bg; each cell has an inner sprite pattern.
- Each row of cells has exactly one cell with a unique pattern; the rest are copies.

Rule:
- For each lattice row, find the cell whose inner pattern appears exactly once (the "odd one out").
- Construct a single-column lattice preserving the original row heights, containing only the unique cells.

Key insight: The task is per-row "odd one out" — every cell in a row is identical except one oddball, which must be extracted.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="62593bfd" lines="32-46">
<description>
Sort objects to top or bottom edge by whether their shape (up to rotation/flip) is shared.

Input structure:
- bg = most common color.
- Several multicolor connected components (8-connected) scattered across the grid.

Rule:
- Compute each object's canonical shape (binary mask, invariant to rotation and flip).
- Objects whose shape appears in two or more components → translate to TOP edge (row 0).
- Objects whose shape is unique → translate to BOTTOM edge (row H−h).
- Horizontal position (left column) is preserved; only vertical translation changes.

Key insight: The sorting criterion is shape frequency (rotation-invariant), not area — duplicated shapes go up, unique shapes go down.
</description>
</reference_puzzle>

你的任务：
以参考题为标杆，运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。用中文创造全新的 ARC 谜题描述。
创新方式包括但不限于：
**机制重组**：将多道参考题各自的核心机制提取出来，组合成一个新的多步管道，用到新的场景。
**机制变异**：取几道参考题的核心机制，以机制为基础进行创新并与其他多种机制进行创新重组，使解题路径发生质变而非量变。
**逆向构造**：取一道参考题的输出模式，思考"什么样的输入规则能产生类似的输出结构但需要完全不同的推理过程"。

严格使用下面的 XML 风格格式输出：

<new_puzzle>
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。
</new_puzzle>

请一次性生成 5 个 `<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 5 个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7. 题目难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
8. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
9. 题目的难度优先来自规则的**结构层数**，而不是单一步骤的复杂度。关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。以两层结构为例：
   - ✗ 伪两层："找到缺口方向"→"把东西推到对面" — 第一步肉眼可见，实际只有一层推理。
   - ✓ 真两层："通过孔洞数量匹配颜色图例"→"用匹配到的颜色重绘对象" — 两步都需要非平凡的归纳。

   如果你发现自己写出的题只是"对每个对象，根据它的某个属性做某变换"的单步规则，请引入一个需要主动推理的前置步骤。**用堆叠 tie-break 规则或复杂数学公式来"加难度"是劣解；用多一层需要归纳的结构嵌套来"加难度"才是好解**。
10. 在符合ARC题目和提示词要求的基础上，鼓励尝试创新规则为多层结构的题目，不局限于两层，可以创新三层或四层结构的题目，但不可为了多结构层数违背要求。

生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。没有使用上述注意事项第二条禁止使用的复杂计算。

