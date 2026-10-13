你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="b7256dcd" lines="18-32">
<description>
Transfer each colored marker to its adjacent neutral mask component, then erase the markers.

Input structure
- The background is the most common color.
- The neutral mask color is the unique most frequent foreground color.
- Other foreground colors are marker pixels that may touch neutral 4-connected mask components.
Rule
- Remove every marker pixel by recoloring it to the background.
- For each 4-connected neutral mask component, find marker colors touching it by edge adjacency.
- If a component touches one marker color, recolor the whole neutral component to that marker color.
- If a component touches no marker color, leave it neutral.
Key insight
- Marker pixels label neighboring neutral regions but are not part of the final colored objects.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="f8f52ecc" lines="107-122">
<description>
Connect same-colored foreground components with one-cell-wide orthogonal paths around teal wall segments.

Input structure
- The dominant color is the background.
- Teal cells are fixed wall segments and are never source connectors.
- Other foreground colors appear as one or more 4-connected components; colors with a single component stay unchanged.

Rule
- For each non-background, non-teal color, read its 4-connected components in row-major order.
- Connect each consecutive pair by the shortest clear L-shaped path between the two components.
- Clear paths may pass only through background or the same color, and they preserve the previous travel axis when both bends are possible.

Key insight
- The output is not a global recolor: each color traces a routed chain through its own components while teal cells and other colors act as blockers.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="f28a3cbb" lines="42-50">
<description>
Compress each colored ray pixel inward until it touches that color's 3x3 anchor block.
Input structure
- A uniform background grid with non-background colors.
- Each foreground color contains one solid 3x3 anchor block; all other pixels of that color lie on straight 8-neighbor rays aimed at that block.
Rule
- Keep anchor blocks fixed, erase loose foreground pixels, and for each ray pack its pixels into consecutive cells adjacent to the anchor along the same ray.
Key insight
- The scattered pixels are displaced pixels on visible row, column, or diagonal sightlines back to their same-colored 3x3 block.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="f18ec8cc" lines="71-86">
<description>
Reorder the vertical color-major blocks while preserving every block's internal pixels.

Input structure
The grid is divided into vertical blocks, each with a unique dominant color.
Small marker pixels from other block colors may appear inside the blocks.

Rule
If adjacent shared-color marker rows read exactly as the current block order
from right to left, reverse the blocks. Otherwise move the leftmost block to
the far right and shift the remaining blocks left.

Key insight
The transformation acts on whole vertical blocks, so marker pixels move with
their source block instead of being recolored or independently rearranged.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="d59b0160" lines="39-54">
<description>
Delete connected components whose colors match a legend swatch.

Input structure:
- Background (bg) fills most of the grid.
- Top-left corner: a small square legend region bordered on its bottom and right edges by a single color; the interior holds a set of "target" colors.
- Elsewhere: multicolored objects (4-connected components) on the bg.

Rule:
- Read the target color set from inside the legend border.
- For each connected component, check whether it contains **all** target colors.
- Erase (fill with bg) every component that does; leave all others unchanged.
- Output grid is the same size.

Key insight: The legend acts as a filter specification — only components carrying the full color palette shown in the legend are removed.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="271d71e2" lines="9-22">
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

<reference_puzzle index="7" source="train" puzzle_id="1b8318e3" lines="86-105">
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

<reference_puzzle index="8" source="train" puzzle_id="f0100645" lines="51-68">
<description>
Slide each inner colored object horizontally into the same-colored vertical wall until it touches an obstacle.

Input structure
- The background is the dominant color.
- The only foreground colors are two non-background colors.
- One foreground color fills the left column and the other fills the right column.
- Extra 8-connected objects of each wall color appear away from that color's wall.

Rule
- Clear the extra objects from their original positions.
- For each wall color, process its 8-connected extra objects from nearest to farthest from the matching wall.
- Slide each object horizontally toward its matching wall as far as possible without overlapping the wall or an already settled object of the same color.
- Keep the two vertical wall columns fixed.

Key insight
- The side columns act like color-specific gravity walls: same-colored objects compact sideways against their own wall while preserving each object's shape.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="8886d717" lines="43-63">
<description>
Repair markers embedded in the main object and copy the remaining markers one pixel toward the solid edge.

Input structure:
- A mostly background grid contains one solid non-background edge line.
- Apart from that edge and the background, there is one large main object
  color and a smaller marker color.
- Each marker is either embedded in the main object, touching it on at
  least three 4-neighbor sides, or isolated on the background.

Rule:
- Embedded markers are recolored to the main object color.
- Isolated background markers remain in place and also paint the adjacent
  background pixel in the direction of the solid edge line.
- All other pixels are unchanged.

Key insight:
The full edge line is a direction cue: it tells which way the background
markers extend, while markers surrounded by the main object are simply
defects to be filled.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="dc46ea44" lines="11-25">
<description>
Move the lower sprite and its pink guide into the upper panel, aligning the sprite to the guide's lower-left branch.

Input structure
- The grid has one dominant background color and one full-width separator row.
- Below the separator are exactly two foreground roles: a five-pixel guide marker and one monochrome sprite.
- The guide marker has normalized shape {(0,2), (1,2), (2,0), (2,2), (3,1)} and starts immediately below the separator.

Rule
- Clear the canvas to the background, preserving the separator row.
- Translate the guide marker upward so its top row becomes row 0.
- Move the sprite, preserving its color and shape, so the sprite bbox bottom-right aligns to the translated guide's lower-left branch.

Key insight
- The marker supplies the destination anchor: its lower-left branch cell is where the moved sprite's bbox bottom-right must land.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="84ba50d3" lines="36-54">
<description>
Drop the blue sprites downward relative to the red divider, with vertical sprites passing through to the bottom.

Input structure:
- The background is the dominant color.
- One non-background color forms a single full-width horizontal divider row.
- The other non-background color forms 4-connected sprites above the divider.

Rule:
- Clear the original sprite pixels.
- A one-column sprite falls to the bottom of its column and deletes the divider
  pixel it crossed.
- Any wider sprite falls until its unique widest horizontal row is immediately
  above the divider row; if the moved sprite overlaps the divider, it covers it.

Key insight:
The red row is not a generic floor: only one-column sprites pass through to the
bottom, while wider sprites are registered by their widest horizontal segment.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="d753a70b" lines="81-93">
<description>
Red diamond outlines contract by one layer and gray diamond outlines expand by one layer while other objects stay fixed.

Input structure
- A 16x16 orange-background grid contains one-cell-thick Manhattan-diamond outlines.
- Red and gray outlines are the changing objects; other colored diamond outlines are static.
- Some outlines may be clipped by the grid boundary, but their visible pixels still fit a clipped Manhattan-distance contour.
Rule
- Clear all red and gray pixels.
- Red objects are redrawn with the same diamond center and radius minus one.
- Gray objects are redrawn with radius plus one; an ambiguous edge-clipped growing arc continues away from the nearest perpendicular edge.
Key insight
- The task is not about moving whole objects: each changing object is a contour of a diamond, and the visible operation is one step of contour contraction or expansion.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="7b5033c1" lines="7-21">
<description>
Straighten a colored snake path into a single-column strip preserving pixel order.

Input structure:
- bg = most common color.
- Non-bg pixels form a single 4-connected path (snake) with exactly two endpoints.
- Each path pixel may be a different non-bg color.

Rule:
- Find the two endpoints (cells with only one path neighbor) of the snake.
- Begin at the top-left-most endpoint and trace the path cell by cell.
- Output an (N×1) column listing each cell's color in traversal order.

Key insight: The path is a simple chain with no branches; choosing the top-left-most endpoint fixes a unique reading direction.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="37ce87bb" lines="33-47">
<description>
Compute a grey result bar by adding teal bar heights and subtracting red bar heights.

Input structure
- The grid has a dominant background and several bottom-aligned one-cell-wide vertical bars.
- Teal bars are positive operands; red bars are negative operands.
- The penultimate column is an empty result slot.

Rule
- Keep every input bar unchanged.
- Compute result height = total teal bar height - total red bar height.
- Paint that many bottom cells of the penultimate column grey.

Key insight
- The bars encode arithmetic by height, with color providing the sign.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="3ad05f52" lines="128-142">
<description>
Paint the maze corridor and framed rooms lying on shortest paths between the two colored endpoint components.

Input structure
- The grid has a dominant background, one wall color forming thin orthogonal frames, and one endpoint/fill color.
- The endpoint/fill color forms exactly two 4-connected solid rectangular components; their larger side lengths define the framed room sizes.

Rule
- Treat wall pixels as blocked and every other pixel as passable.
- Find every passable pixel that lies on a shortest 4-connected path between the two endpoint components.
- Also fill any framed square room of an endpoint-derived size that intersects that shortest-path mask.

Key insight
- Shortest paths identify the route through open connector corridors, while the wall-bordered square rooms need to be expanded from the route centerline to their full interior.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="eee78d87" lines="40-55">
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

<reference_puzzle index="17" source="train" puzzle_id="412b6263" lines="39-54">
<description>
Infer the background, rotate the whole input 90 degrees counterclockwise, and stack two framed copies vertically.

Input structure:
- Blue is absent from the input.
- One rare marker color appears in exactly two pixels.
- The background is the unique non-marker color whose removal leaves exactly two 4-connected foreground objects, each with one marker pixel and one body color.
- The whole grid, including background and foreground objects, is the sprite to transform.

Rule:
- Rotate the entire input grid 90 degrees counterclockwise.
- Create a blue one-pixel frame around two vertical copies of the rotated grid.
- Use the input background color for the two endpoints of every horizontal frame/separator row.

Key insight: The output is not a crop of the foreground objects; it is two copies of the complete rotated input, separated and enclosed by a newly added blue border.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="98c475bf" lines="62-76">
<description>
Erase the complete interior glyph and expand each endpoint marker pair into its color's full glyph.

Input structure
- The grid is 20x20 with background inside two same-color full-height vertical frame bars.
- Inside the frame there is one complete color glyph and one or more other colors appearing only as two endpoint markers in the same row.
- Marker endpoints are always in the first and last interior columns.

Rule
- Keep the vertical frame unchanged.
- Erase any complete interior glyph.
- For each endpoint marker pair, use that marker color's learned glyph template and anchor its horizontal bar on the marker row.

Key insight
- Color identifies which glyph shape to draw; the two endpoint pixels are not copied objects, they are anchors for reconstructing that color's full symbol.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="b10624e5" lines="24-39">
<description>
Replicate a motif across quadrants with scaling and reflection keyed to red anchor blocks.

Input structure:
- Grid divided into four quadrants by a full-row and full-column divider of a single non-bg color.
- Each quadrant contains a red (2) square anchor block of varying size.
- The top-left quadrant also has a colored motif attached to its red anchor.

Rule:
- Use the top-left quadrant's motif as the source pattern.
- For each other quadrant, scale the motif by the ratio of that quadrant's red block size to the source red block size.
- Reflect horizontally for right-side quadrants, vertically for bottom quadrants (both for bottom-right).
- Align the scaled/reflected motif so its red region overlaps the target anchor, then stamp non-bg pixels.

Key insight: Red block size encodes the scale factor; quadrant position encodes the reflection axes.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="84551f4c" lines="42-53">
<description>
Rotate each blue-started chain of bottom-aligned vertical bars into horizontal bars resting on the same baseline.
Input structure
- The grid has a background and bottom-aligned, one-cell-wide vertical bars.
- Blue bars are markers; red bars are body bars that may be chained after a marker.
Rule
- Starting at each blue bar, repeatedly include the next red bar whose column is exactly the current bar's column plus the current bar's height.
- Clear every included vertical bar, then redraw each included bar horizontally to the right from its original bottom cell, with length equal to its vertical height.
- Red bars that are not reached from a blue marker remain unchanged.
Key insight
- A vertical bar's height is also the width it occupies after rotation, so the chain follows immediate horizontal contact after each rotated bar.
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
