你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="458e3a53" lines="19-32">
<description>
Extract the compact table of solid cells hidden inside a separator lattice.

Input structure
- The grid has full-width and full-height bars of one separator color, dividing it into rectangular cells.
- Most cells contain a repeating multicolor pattern, while a contiguous rectangle of cells is solid monochrome.

Rule
- Split the input into lattice cells using the full separator rows and columns.
- Keep the cells whose full interior is one color, crop to their occupied rectangle, and output those colors.

Key insight
- The periodic pattern is noise; only completely solid lattice cells encode the reduced output grid.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="42f14c03" lines="75-83">
<description>
Crop the foreground object whose missing cells match marker shapes, and fill those missing cells with the marker colors.

Input structure
A dominant background surrounds several monochromatic foreground objects. Exactly one object has background gaps inside its bounding box whose component shapes are matched by separate marker objects.
Rule
Find that object, crop its bounding box, and replace each background gap component with the color of the separate marker component or components having the same normalized shape.
Key insight
The marker objects do not mark absolute positions; their shapes identify which gaps in the selected object should be colored, while objects with unmatched gaps are distractors.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="7b80bb43" lines="8-21">
<description>
Remove diagonal staircase noise and complete broken straight lines.

Input structure:
- bg = background; one foreground color fg forming axis-aligned lines.
- Some fg pixels form 45° diagonal chains (8-connected but 4-disconnected from lines).

Rule:
- Delete all diagonal-only fg pixels (those with no orthogonal fg neighbor, only diagonal ones).
- Fill short gaps in horizontal/vertical lines where a deleted diagonal was nearby.
- Prune remaining diagonal-adjacent leaf pixels near former diagonal sites.

Key insight: Diagonal pixels are hints marking where a straight line is broken; their span sets the maximum gap size to fill.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="981add89" lines="10-29">
<description>
Extend each top-row control pixel downward as a vertical toggle ray in its own color.

Input structure:
- The background is the dominant color.
- Non-background pixels on the top row are control pixels, each identifying
  one column and one ray color.
- Foreground regions below may intersect these control columns.

Rule:
- For every top-row control pixel, process the cells below it in the same
  column down to the bottom edge.
- A cell that already has the control color is erased to the background.
- Every other cell on that downward ray is painted with the control color.
- The top-row control pixels themselves remain unchanged.

Key insight:
The top row is a set of vertical ray launchers; each ray toggles only cells
matching its own color and overwrites all other colors along that column.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="66ac4c3b" lines="57-72">
<description>
Complete the sparse-color reflected copy of the foreground motif across its matching marker row or column.

Input structure
- The grid has one background color and two foreground colors.
- One foreground color appears only as a sparse marker row or column.
- The other foreground color contains the same sparse marker positions on a parallel row or column, plus the motif on the far side of the sparse markers.

Rule
- Keep the original grid unchanged.
- Reflect every motif pixel of the larger foreground color across the two parallel marker lines.
- Paint the reflected motif pixels with the sparse marker color.

Key insight
- The two equal marker rows or columns are paired landmarks: the sparse-color marker line borders the source motif, and the matching source-color marker line anchors the reflected copy.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="d749d46f" lines="32-51">
<description>
Pack each solid foreground rectangle twice: wide across the top and tall from the bottom.

Input structure
The grid has one background color and one foreground color; the foreground forms
separate solid rectangles, ordered from left to right.

Rule
Build a 10-row background canvas whose width is the packed width of the wide
versions of the rectangles. For each source rectangle, paint a top-aligned
rectangle of shape min(h,w) by max(h,w), with one background column between
consecutive rectangles. Then, starting again from the left edge, paint a
bottom-aligned rectangle of shape max(h,w) by min(h,w) for each source rectangle,
again separated by one background column.

Key insight
The input rectangles are not copied in their original orientations; each object
contributes one horizontal-facing and one vertical-facing normalized rectangle,
preserving left-to-right object order.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="4c416de3" lines="99-118">
<description>
Replicate a pre-drawn corner pattern to every other marker-bearing corner.

Input structure:
- bg = most frequent color. Hollow 0-frames (possibly partial L-shapes); each
  frame-corner cell is either 0 or a painted (non-bg, non-0) color.
- >= 1 'source' corner: the pattern is already drawn (one color, both outward
  cells and an inward-diagonal trail).
- Other corners: a single colored pixel on the inward diagonal at the pattern's
  deepest (k,k) offset; the pixel's color is that corner's target color.

Rule:
- Extract the canonical (TL-oriented) pattern from any source.
- For each non-source corner whose marker cell is colored, paint the same
  pattern rotated to that corner's orientation, using the marker's color.

Key insight: the pattern is symmetric about the corner's inward-diagonal axis
(canonical dr == dc), so it reflects cleanly across all four corner orientations.
Painting uses each target's OWN marker color, not the source color.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="1b8318e3" lines="86-105">
<description>
Move each isolated colored pixel into the one-cell frame around a grey 2x2 block.

Input structure:
- Background is the dominant color.
- Grey foreground objects are solid 2x2 anchor blocks.
- Every other non-background, non-grey cell is a colored singleton source.

Rule:
- Keep the grey anchors fixed and clear all colored singleton source pixels.
- For each singleton and each anchor, the normal destination is the cell reached by
  clamping the pixel into the anchor's one-cell outer frame.
- If a diagonally outside singleton has an exact 8-direction sightline into an anchor,
  also allow the cell just before the first grey pixel on that ray.
- Choose distinct destinations that minimize total king-move distance; at equal
  distance prefer exact sightlines, then the shorter Manhattan movement.

Key insight: The grey 2x2 blocks define magnetic frames; conflicts are resolved
structurally by nearest-frame assignment rather than by color identity.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="581f7754" lines="8-22">
<description>
Slide multi-color sprites to align their marker pixels with matching isolated border anchors.

Input structure:
- bg = most common color.
- Multi-color sprites in the interior; each has a minority-color "marker" pixel.
- Isolated border pixels (non-bg with all 4-neighbors bg) act as anchors.

Rule:
- Match each sprite to the border anchor sharing its marker color.
- Top/bottom-edge anchor (non-corner): slide sprite horizontally so marker column = anchor column.
- Left/right-edge anchor (or corner): slide sprite vertically so marker row = anchor row.

Key insight: The anchor's edge position determines the slide axis — top/bottom means horizontal, left/right (or corner) means vertical.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="9f41bd9c" lines="10-30">
<description>
Move the side-attached foreground object to the opposite wall, shear its lower rows inward, and paint the crossed floor cells.

Input structure:
- The top row is the upper background color and the bottom row is the
  floor color.
- A single non-background, non-floor object touches exactly one vertical
  grid edge and rests immediately above the first solid floor row.

Rule:
- Clear the source object to the upper background color.
- Repaint each object row on the opposite side: the first three relative
  rows keep the edge alignment, and each lower row shifts one more pixel
  inward from that side.
- On the first floor row, repaint the traversed side-to-object segment
  maroon, stopping just before the final outer bottom support pixel.

Key insight:
- The side of the source object gives the horizontal direction, while the
  row index inside the object gives the one-cell-per-row diagonal shear.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="1b59e163" lines="54-69">
<description>
Copy each marked sprite to every standalone marker of the same color and erase the original sources.

Input structure:
- The background is the dominant color.
- Each source is an 8-connected component containing a monochrome sprite and one marker pixel of another color.
- Other isolated marker pixels may be destinations; marker colors with no source sprite are distractors.

Rule:
- Clear the grid to background.
- For every source marker color, place the sprite at each other marker of that color using the source marker as the anchor.
- Keep each destination marker pixel in place; do not keep source sprites, source markers, or distractor markers.

Key insight:
- A colored marker is not part of the sprite geometry; it is the anchor that says where same-colored copies of that sprite should be stamped.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="b2bc3ffd" lines="8-23">
<description>
Move every object above the solid bottom stripe upward by its own pixel count.

Input structure
The grid has a dominant background, exactly one full-width non-background row
that acts as a fixed stripe, and one or more 4-connected monochrome objects
above that stripe.

Rule
Clear each object from its original cells, then translate it straight upward
by the number of pixels in that object, preserving its shape and color. Leave
the full-width stripe unchanged.

Key insight
Each object's area is also its vertical displacement; the full-width row is a
fixed baseline, not a moving object.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="8618d23e" lines="9-25">
<description>
Stagger the two horizontal input regions onto a larger maroon separator canvas.

Input structure:
- The input is a rectangular grid with an even number of rows.
- The top half and bottom half are the two regions to preserve.
- Maroon is absent from the input and is used only as the new separator color.

Rule:
- Create an output with one extra row and one extra column, filled with maroon.
- Copy the top half into the upper-left corner.
- Copy the bottom half one row below the top region and one column to the right.

Key insight:
The output is not a recoloring of either region; it inserts an L-shaped maroon
gap by diagonally offsetting the lower half from the upper half.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="42f83767" lines="23-39">
<description>
Expand the lower color grid by replacing each pixel with its matching top stencil.

Input structure
The input has a top legend and a lower dense color grid, separated by blank rows.
The legend contains left-to-right 2x2 color swatches and, to their right,
the same number of square neutral-color stencil masks.

Rule
Pair the swatches and stencil masks by left-to-right order. For every pixel in
the lower color grid, paste the paired stencil mask as one output block,
recoloring the mask pixels with that pixel's color and leaving mask holes as bg.

Key insight
The lower grid is not copied directly; it is an index grid whose colors select
the stencil shape from the legend, so each input pixel becomes one square block.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="195c6913" lines="37-52">
<description>
Simulate particles along a path with alternating turn direction at collisions.

Input structure:
- Top-left: 2x2 monochrome blocks define a repeating color cycle (left-to-right).
- Isolated single pixels of a cycle color: start markers adjacent to the path.
- A 2x2 monochrome block of a distinct color: end marker.

Rule:
- Each start marker launches a particle moving straight, painting cycle colors in order.
- On hitting a boundary, paint that cell with the end color and turn 90°.
- Turn direction alternates: odd collisions → left (CCW), even → right (CW).
- If the scheduled turn is blocked, the particle stops.

Key insight: The alternating left/right turn rule is the core mechanism; without it the path would be trivial.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="9968a131" lines="10-27">
<description>
Shift alternating two-color row bars so each row aligns to the topmost bar's color columns.

Input structure:
- The dominant background surrounds horizontal two-pixel bars.
- Every foreground row has exactly one adjacent bar using the same two colors.
- The topmost bar gives the reference left/right column for each color.

Rule:
- Keep bars whose left color is already in its reference column.
- For a reversed bar, shift the whole two-pixel bar horizontally until its
  left color lands in that color's reference column.
- Background cells vacated by shifted bars become background.

Key insight:
- The rows are not recolored; each two-color bar is translated so equal
  colors line up with the topmost reference bar.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="221dfab4" lines="6-19">
<description>
Propagate a repeating striped pattern from a yellow border segment toward the opposite edge, recoloring foreground objects on certain stripes.

Input structure:
- bg = most common color. Foreground blobs of another color scattered in the interior.
- A short yellow (4) segment sits on one border edge.

Rule:
- From the yellow segment, march toward the opposite edge in steps of 2 cells, cycling colors Y, Y, G (4, 4, 3). The 1-cell gap between stripes is forced to bg.
- Yellow stripes paint only within the segment's span.
- Green stripes additionally recolor all non-bg pixels across the entire row/column to green.

Key insight: Green stripes act as a "flood" that recolors existing objects on the full row/column, while yellow stripes are confined to the segment span.
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

<reference_puzzle index="19" source="train" puzzle_id="6bcdb01e" lines="10-24">
<description>
Extend the edge-touching green start bar through the corridor, turning at each teal blocker until the path leaves the grid.

Input structure:
- The background is the most common color.
- Exactly two foreground colors are present: a 2-pixel 4-connected start bar and many blocker pixels.
- Exactly one pixel of the start bar lies on the grid border.

Rule:
- Start from the border pixel of the bar and move through the other bar pixel into the grid.
- Paint every traversed non-blocker pixel with the start color.
- When the next cell is a blocker, turn to the unique perpendicular non-blocker neighbor; when the next cell is outside the grid, stop.

Key insight: The blocker pixels define a one-cell-wide corridor, so every interior collision has exactly one valid 90-degree turn.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="538b439f" lines="122-142">
<description>
Reflect the solid rectangular source objects across the full separator line and color the gap with the separator color.

Input structure:
- A dominant background contains one full-width or full-height one-pixel
  separator line in a non-background color.
- One other foreground color forms solid rectangular objects, all strictly on the same side of that separator.
- Remaining non-background pixels are scattered distractors.

Rule:
- Keep the input grid shape and unchanged pixels outside each affected corridor.
- Mirror each source object across the separator line.
- For every source object, recolor the rectangular strip between the
  original object and its reflection to the separator color, then paint the
  reflected object in the source color.

Key insight:
- The full straight line is a mirror axis; the real objects are identified
  structurally as the only solid rectangles confined to one side, while
  scattered pixels are noise that may be overwritten inside the corridor.
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
   - **不建议使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id是不可用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
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
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。
