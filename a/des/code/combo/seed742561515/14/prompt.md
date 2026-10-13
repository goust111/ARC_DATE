你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="52df9849" lines="49-67">
<description>
Complete each partially hidden simple foreground shape by recoloring the occluding pixels in its footprint.

Input structure
- The grid has a dominant background and several solid foreground shapes.
- Some shapes are partially hidden by other foreground colors, but their full
  footprint is still indicated by a foreground-filled bbox or by a foreground-
  filled 45-degree diagonal line.

Rule
- For each foreground color, complete its full rectangular bbox when that bbox
  contains no background and includes other colors.
- Otherwise, complete its 45-degree diagonal line when all visible pixels lie
  on that line and the hidden line cells are foreground, not background.
- Recolor only those hidden footprint pixels; leave all other pixels unchanged.

Key insight
- The occluded cells are exactly the non-background cells inside a simple
  shape's implied footprint, so completing shapes is a local recoloring problem.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="470c91de" lines="48-65">
<description>
Move each teal-cornered filled rectangle one cell diagonally toward its teal corner.

Input structure:
- Dominant background surrounds several filled rectangular colored objects.
- Each object has exactly one teal control pixel replacing one corner cell of
  its rectangle.

Rule:
- Treat the teal pixel as the missing corner of its colored rectangle.
- Erase each original rectangle footprint, including the teal pixel.
- Repaint the completed rectangle shifted one cell diagonally in the direction
  of that teal corner.

Key insight:
- The control pixel's corner position gives both the missing cell to restore
  and the diagonal motion vector.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="e3fe1151" lines="36-55">
<description>
Fill the one blank in each corner region so all four regions have the same color multiset.

Input structure:
- The dominant background color forms one solid separator row and one solid
  separator column.
- The separator cross splits the grid into four equal rectangular corner regions.
- Each corner region contains exactly one background-colored blank pixel.

Rule:
- Count the non-background colors in each corner region.
- The completed region multiset is defined by the maximum observed count of
  each color across the four regions.
- Replace each region's blank pixel with the single color needed to match that
  completed multiset.

Key insight:
- The separator color inside a corner region is a missing value marker; comparing
  the four regions reveals the shared multiset without using numeric color order.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="9968a131" lines="10-27">
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

<reference_puzzle index="5" source="eval" puzzle_id="5dbc8537" lines="10-23">
<description>
Fill rectangular cavities in a template shape by placing source sprites into matching holes.

Input structure:
- bg = most common color. A dominant "base" color forms a large frame/template with rectangular gaps.
- Small colored sprites sit outside the template region.

Rule:
- The base color (largest connected component excluding bg) defines the template bbox.
- Each rectangular hole inside the template matches exactly one outside sprite by size.
- Place each sprite into its matching cavity; output is the filled template bbox.

Key insight: Sprite bounding boxes match cavity dimensions exactly — it's a jigsaw-fit puzzle where each piece has a unique slot.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="458e3a53" lines="19-32">
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

<reference_puzzle index="7" source="train" puzzle_id="985ae207" lines="82-97">
<description>
Repeat each 3x3 control-center sprite toward the matching solid rectangle until a copy first overlaps that rectangle.

Input structure
- The background is the dominant color.
- Each sprite is a 3x3 foreground object whose eight border pixels have one color and whose center has a different control color.
- For every control color, there is exactly one separate solid rectangular object of that color.

Rule
- Match each sprite to the unique solid rectangle whose color equals the sprite center.
- Copy the whole 3x3 sprite in the cardinal direction of that rectangle, stepping by three cells each time.
- Stop after pasting the first copy whose 3x3 bbox overlaps the rectangle; copied pixels overwrite the rectangle there.

Key insight
The center pixel is not the paint color: it is a destination key, while the entire 3x3 sprite is the motif tiled along the straight path.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="2de01db2" lines="28-41">
<description>
Invert each row's native foreground color against the black background.

Input structure
- A black-background grid whose rows each have one native foreground color plus optional distractor colors.
- The native row color is the most frequent non-bg color in that row; if tied, its run starts leftmost among tied colors.

Rule
- For each row independently, turn pixels of the native row color into bg.
- Turn every other pixel in that row, including bg and distractor colors, into the native row color.

Key insight
- Distractor colors are not preserved objects; they mark cells that participate in the binary complement of the row's native color.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="89565ca0" lines="120-136">
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

<reference_puzzle index="10" source="train" puzzle_id="5e6bbc0b" lines="42-56">
<description>
Compact the checkerboard's moving color toward the edge marked by the teal pixel, leaving a maroon swept trail on the marker line.

Input structure
- A blue/black checkerboard with exactly one teal marker replacing a blue pixel on a non-corner edge.
- The marker's checkerboard parity identifies the moving color (blue); the opposite parity is background (black).

Rule
- If the marker is on the left or right edge, compact moving-color pixels in every row toward that side.
- If the marker is on the top or bottom edge, compact moving-color pixels in every column toward that side.
- Preserve the teal marker at its edge cell.
- On the marker's row or column, color cells swept by shifted moving pixels maroon unless the final compacted line occupies them.

Key insight
The marker selects both the compression direction and the checkerboard color class that moves; maroon records the movement trail only along that marked line.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="34cfa167" lines="94-117">
<description>
Complete a four-corner framed rectangle from one decorated anchor rectangle and its diagonal mate.

Input structure:
- The background is the dominant color.
- Two same-color solid rectangular anchors of the same size sit on opposite
  diagonal corners, with the first anchor above-left of the second.
- The upper-left anchor has two one-cell-thick vertical control bars to its
  right and two one-cell-thick horizontal control bars below it.

Rule:
- Copy the anchor rectangle into the two missing corners of the implied
  larger rectangle.
- Repeat the right-side control bars horizontally between the left and right
  anchors, using their spacing as the period, and draw the outer top and
  bottom borders in the first right-control color.
- Repeat the below-side control bars vertically between the top and bottom
  anchors, using the same period, and draw the outer left and right borders
  in the first below-control color.

Key insight: The small control bars are the first two ticks of two periodic
frame patterns; the diagonal anchor supplies the opposite corner and the
missing two anchor copies.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="18286ef8" lines="17-31">
<description>
Use the pink control pixel to choose where the maroon pixel belongs inside the grey 3x3 region, and turn the control pixel maroon.

Input structure:
- The grid contains one 3x3 region made from eight grey pixels and one maroon pixel.
- There is exactly one pink control pixel outside that region.

Rule:
- Restore the old maroon cell in the 3x3 region to grey.
- Compare the pink pixel to the 3x3 region: above/same band/below selects top/middle/bottom, and left/same band/right selects left/middle/right.
- Paint the selected cell maroon and recolor the pink control pixel maroon.

Key insight:
- The pink pixel acts as a relative-position marker for a 3x3 target inside the grey region.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="9f5f939b" lines="42-57">
<description>
Mark the missing center of each plus-shaped arrangement of four two-pixel bars with yellow.

Input structure:
- The background is the dominant color.
- The single foreground color forms isolated 4-connected objects, each a horizontal or vertical two-pixel bar.
- Some bars form an implied plus sign: two horizontal bars and two vertical bars around an empty center, with a one-pixel gap between the center and each bar.

Rule:
- Keep the input unchanged except at implied plus-sign centers.
- For every background pixel with foreground bars at offsets 2-3 cells left, right, above, and below, recolor that center yellow.
- The one-pixel gaps immediately next to the center in the four cardinal directions must remain background.

Key insight:
- The yellow pixels are not extending a line; they identify the shared missing center where four separated two-pixel bars align as a plus sign.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="f1bcbc2c" lines="40-55">
<description>
Paint the top-connected middle channel between two orange path components teal.

Input structure
- The background is black.
- One foreground color forms exactly two 4-connected one-pixel-wide path components.
- Optional non-background control pixels may sit in the channel between the two paths.

Rule
- Dilate each path component by one cell in the 8-neighborhood and intersect the two dilations; this gives the one-cell-wide middle channel.
- A control pixel in that channel is a blocker only when its 4-neighbors touch fewer than both path components; otherwise it is part of the channel and is repainted.
- After removing blockers, paint the channel component connected to the top edge teal and keep all other pixels unchanged.

Key insight
- The desired pixels are not found by region filling; they are the geometric overlap of the two paths' one-cell halos, with weakly attached control pixels splitting the channel.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="e84fef15" lines="39-59">
<description>
Reconstruct the repeated tile from its separated copies and mark the missing pixels in the damaged copy blue.

Input structure:
- The input is a lattice of equal-size blocks separated by one-cell solid rows
  and columns of the same separator color.
- All blocks are copies of one motif except exactly one block, where some
  motif pixels have been overwritten by the motif background color.
- Blue is absent from the input and is reserved as the output marker color.

Rule:
- Recover the motif by taking the unique majority color at each local block
  coordinate across all block copies.
- Find the single block that differs from this consensus motif.
- Return the consensus motif, recoloring the differing local coordinates blue.

Key insight:
- The repeated copies provide redundancy: the damaged positions are exactly
  the coordinates where the lone nonconforming copy has background instead of
  the consensus motif color.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="67e490f4" lines="8-22">
<description>
Fill holes inside a large template frame using shape-matched outside sprites.

Input structure:
- bg = most common color. One large monochrome frame object (template).
- Inside the frame: holes of various shapes (non-frame-color regions).
- Outside the frame: small monochrome sprites scattered on bg.

Rule:
- Crop to the frame's bounding box; fill entirely with frame color.
- For each hole, find an outside sprite whose shape matches (rotation/flip allowed).
- Fill the hole with the matched sprite's color. If multiple sprites match, use the most frequent color.

Key insight: Sprites act as shape-coded color keys — their shape identifies which hole they fill, their color determines the fill color.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="5034a0b5" lines="16-32">
<description>
Shift each interior pixel whose color matches a frame side one cell toward that matching side.

Input structure
The grid has a one-cell-thick rectangular frame: the four corners are the
background color, and each side between corners is a solid, distinct side
color. The interior is mostly background with sparse foreground pixels.

Rule
Interior pixels with the top-side color move one row up, bottom-side color
move one row down, left-side color move one column left, and right-side color
move one column right. A pixel already adjacent to its matching side stays in
place; colors that do not match a frame side and the frame itself are unchanged.

Key insight
The colored frame sides act as direction labels for matching interior pixels.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="46c35fc7" lines="21-34">
<description>
Rotate the colors around every 3x3 foreground ring, with corners and side-centers cycling in opposite directions.

Input structure
- The grid has a dominant background color.
- Each 4-connected foreground object is a complete 3x3 ring: eight foreground pixels around a background center.

Rule
- For each ring, move corner colors one step around the four corners.
- Move side-center colors one step around the four side centers in the opposite direction.
- Leave the background center and all other background pixels unchanged.

Key insight
- Corners and side centers form two separate four-cycles on the same 3x3 ring.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="dd6b8c4b" lines="9-22">
<description>
Move the closest reachable maroon pixels into the 3×3 block around the unique red center.

Input structure:
- bg = orange floor, magenta walls, scattered maroon (9) pixels.
- A unique red (2) pixel marks the center of a 3×3 destination block.

Rule:
- Maroon pixels connected to the destination through floor/maroon cells (not blocked by walls) are candidates.
- Rank candidates by distance to the destination; the closest ones fill the block in row-major order.
- Moved maroons are erased (replaced with floor color).

Key insight: Wall barriers restrict connectivity — only maroon pixels reachable via flood-fill through the floor qualify, so proximity alone isn't enough.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="83eb0a57" lines="72-86">
<description>
Crop the largest object and overlay each smaller sprite where its shared marker pattern matches.

Input structure
- The background is the most common color, and the foreground has one largest destination object plus smaller source sprites.
- Each source sprite shares exactly one marker color with the destination object.

Rule
- Output the tight bbox crop of the largest object.
- Translate each source sprite so every connected marker component of the shared color exactly matches destination marker components.
- Paste source sprites in decreasing area order, so smaller sprites can sit on top of larger ones when they overlap.

Key insight
- The shared-color marker components are alignment keys; matching whole marker components distinguishes singleton slots from larger same-color marker blocks.
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
