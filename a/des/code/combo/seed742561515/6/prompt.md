你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="54dc2872" lines="52-69">
<description>
Move each marked L-shaped sprite onto the separate pixel matching its marker color.

Input structure
- The background is the dominant color.
- Foreground has 4-connected multicolor sprites and isolated one-pixel controls.
- Each sprite has one repeated body color in an L shape plus one embedded marker
  pixel of a different color.

Rule
- If the embedded marker color also appears as a separate isolated control pixel,
  translate the whole sprite so the body L-bend lands on that control pixel.
- Clear each moved sprite from its old position, clear its consumed control pixel,
  then paste all translated sprites; sprites without a matching control stay put.

Key insight
- The marker color pairs a sprite with its destination control, while the L-bend
  gives the orientation-independent anchor cell to place on that destination.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="25e02866" lines="14-31">
<description>
Merge same-sized filled rectangular objects by overlaying their internal marker pixels.

Input structure:
- A dominant background surrounds two or more separated 4-connected foreground objects.
- Every object is a filled rectangle of the same size.
- Each rectangle has the same majority base color plus one or more non-base marker pixels.

Rule:
- Create one output rectangle with the shared object size, filled with the shared base color.
- For each input rectangle, copy every non-base marker pixel to the same relative location
  in the output.
- Marker overlays must either land on base color or agree in color with an existing marker.

Key insight:
The separated rectangles are layers of the same template; their colored marker pixels
combine by local coordinates while the repeated base fill is just the output canvas.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="b9e38dc0" lines="7-18">
<description>
Fill the region enclosed by a wall sprite and radiate color outward through its opening.

Input structure:
- bg = most common color. Wall = most frequent non-bg color, forming a partial enclosure (e.g. U-shape).
- A fill-color pixel sits inside the enclosed area. Other non-bg pixels may act as obstacles.
Rule:
- Flood-fill the enclosed area with the fill color, radiating outward through the wall's open end.
- Obstacles cast "shadows" along the radiate direction, blocking fill behind them.

Key insight: The wall endpoints define the opening direction; fill radiates outward through it, and obstacles block propagation along that same axis.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="cc9053aa" lines="59-72">
<description>
Recolor the unique safe route between the two maroon endpoints from teal to maroon.

Input structure
- A black background contains a teal 4-connected track, exactly two maroon endpoint pixels, and one or more orange blocker pixels embedded near the track.

Rule
- Treat each orange blocker and its 8-neighborhood as forbidden.
- In the remaining teal/maroon track graph, there is exactly one simple
  4-connected path between the maroon endpoints.
- Recolor teal pixels on that path to maroon; leave endpoints, blockers, and all other cells unchanged.

Key insight
- The orange pixels block nearby teal cells diagonally as well as orthogonally, which removes shorter decoy routes and makes the intended endpoint-to-endpoint path unique.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="1efba499" lines="70-85">
<description>
Swap paired marker pixels across the largest foreground-color barrier, placing them just outside the barrier segment.

Input structure:
- The background is the dominant color.
- One foreground color is the unique most frequent non-background color; its pixels form the stationary barrier.
- Other foreground pixels are single markers, sometimes paired on opposite sides of one contiguous barrier segment in the same row or column.

Rule:
- Scan each row and column that crosses exactly one contiguous barrier segment.
- If the line has exactly one marker on each side of that segment, erase both source markers and place each marker in the cell immediately beyond the opposite side of the segment.
- Lines with markers on only one side are unchanged; the barrier remains unchanged.

Key insight:
- The marker pairing is local to a row or column, and the destination is defined by the nearest cells outside the intervening barrier segment, not by the original distance from the barrier.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="ad3b40cf" lines="72-87">
<description>
Mirror the smallest non-axis foreground color across the full straight axis.

Input structure
- The background is the dominant color.
- One foreground color forms exactly one full row, column, main diagonal, or anti-diagonal reflection axis.
- The remaining foreground colors include a unique least-populated color to copy.

Rule
- Preserve the input grid.
- Select every pixel of the unique smallest non-axis foreground color.
- Reflect those pixels across the full straight axis and paint the reflected positions with the same color.

Key insight
- The axis color marks the mirror line, while the color population chooses which set of objects is duplicated.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="bc93ec48" lines="48-63">
<description>
Copy each foreground object anchored in a true grid corner clockwise into the next corner.

Input structure
- The dominant color is the background.
- Foreground objects are monochromatic 4-connected components.
- Relevant objects are exactly those whose bounding box touches two adjacent borders at one grid corner.

Rule
- Start from the input unchanged.
- For each corner object, copy its pixel mask and color to the next corner clockwise: top-left to top-right, top-right to bottom-right, bottom-right to bottom-left, and bottom-left to top-left.
- Preserve each object's orientation by aligning its bounding box to the destination corner; stamps may overwrite existing pixels at the destination, but source pixels are not cleared.

Key insight
- Interior and side-only objects are distractors; only components actually anchored to a corner participate in the clockwise perimeter copy.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="e376de54" lines="41-55">
<description>
Align parallel colored line segments so they all share the same start and end positions as a reference line.

Input structure:
- bg = most common color.
- Several parallel line segments (diagonal, horizontal, or vertical), each a different color, running in the same direction but starting/ending at different positions.

Rule:
- Detect the dominant direction shared by all segments.
- Pick the median segment (by perpendicular offset) as the reference.
- Shift every segment along the shared direction so its start aligns with the reference's start, then trim or extend it to match the reference's length.
- Grid size stays the same; isolated single pixels are untouched.

Key insight: The reference is the median line, not the longest; all segments are snapped to its extent regardless of their original length.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="c3fa4749" lines="79-96">
<description>
Recolor each side-marked multicolor object embedded in a monochrome panel to its marker color.

Input structure:
- Same-color 4-connected components form monochrome panel regions.
- The bounding box of a panel may contain 4-connected non-panel-color objects.
- A target object touches exactly one side of its panel bbox, and all of its
  pixels on that side have one marker color.

Rule:
- For every target object, fill any panel-colored holes enclosed by the object.
- Recolor the object and filled holes to the side marker color.
- Leave all other pixels unchanged.

Key insight:
- A multicolor object's single contact side identifies both which object should
  be normalized and which color should replace it.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="753ea09b" lines="60-79">
<description>
Fill the small background compartments bounded by the long contour color.

Input structure:
- The grid has one dominant background color and two foreground colors.
- The more frequent foreground color forms long 8-connected contour fragments.
- The other foreground color is a fixed distractor/obstacle color that partitions
  the background but is never recolored.

Rule:
- Split the background into 4-connected regions using all foreground pixels as
  barriers.
- Keep the two largest background regions as the outside of the contour drawing.
- Recolor every other background region that touches a long contour component
  to the contour color; leave all foreground pixels unchanged.

Key insight:
The apparent area fill is exactly a component-level fill of bounded background
compartments, not a row-wise interpolation between individual contour pixels.
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

<reference_puzzle index="12" source="train" puzzle_id="6a980be1" lines="46-61">
<description>
Project the framed periodic stripe pattern across the canvas and erase the unused frame cells.

Input structure
- A bg canvas with a one-cell-thick rectangular frame of one color.
- Inside the frame are two parallel full-interior marker bars of another color.
- A third color fills the region between the marker bars on periodically spaced rows or columns.

Rule
- Infer the periodic row or column mask from the third-color stripes inside the frame.
- Paint the frame color along those full rows or columns, extended to the grid boundary.
- Extend the marker bars across the full canvas and repaint the source-color stripe cells between them.

Key insight
- The outer frame is not preserved as a border; it supplies the color for the periodic stripes selected by the inner source pattern.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="142ca369" lines="6-19">
<description>
Trace diagonal rays from triangular arrows, reflecting off colored wall pixels.

Input structure:
- bg grid with monochrome 3-pixel L-shaped "arrows" (2x2 bbox missing one corner) and scattered colored wall pixels.

Rule:
- Each arrow fires a ray diagonally away from its missing corner, painting bg cells in its color.
- A head-on wall hit reverses both axes; a side-touch wall hit flips the touched axis only.
- On any bounce the ray adopts the wall's color and recolors the contact cell.
- Ray stops when it exits the grid or reaches another arrow.

Key insight: Two distinct reflection types (corner vs side-touch) plus color inheritance at each bounce drive the path geometry.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="fbf15a0b" lines="49-69">
<description>
Crop out the grid half containing the two aligned marker pixels, then erase the markers to background.

Input structure
- The grid has a dominant background color and one non-background marker color
  that appears as exactly two cells.
- The marker cells are aligned either in one row or one column and both lie
  in the same half along that direction.
- Other non-background pixels are content inside one or both halves.

Rule
- If the markers share a row, keep the left or right half whose columns
  contain them.
- If the markers share a column, keep the top or bottom half whose rows
  contain them.
- In the kept half, recolor the marker pixels to the background color.

Key insight
- The two marker pixels are not part of the picture; their alignment and
  half-position identify which half of the doubled canvas is the answer.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="58c02a16" lines="10-26">
<description>
Expand the top-left seed mask into a recursive colored pattern and tile it across the full grid.

Input structure
- The background is the dominant color.
- All foreground lies in a top-left rectangle whose last row and last column are a solid secondary color.
- Removing that secondary-color row and column leaves a seed region containing background plus one primary color.

Rule
- Use the primary-color pixels in the seed as a binary mask.
- Replace each primary seed cell with a copy of the seed mask in the primary color, and each background seed cell
  with a copy of the seed mask in the secondary color.
- Repeat and crop that expanded tile to the original grid shape.

Key insight
The solid L-shaped secondary-color border is a legend, not output content; it supplies the color used when the
seed's background cells are expanded.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="5751f35e" lines="29-43">
<description>
Complete sparse nested rectangular color layers into solid rectangles.

Input structure
The background is the dominant color, and every foreground color marks one
axis-aligned rectangular layer; the layer bboxes have unique areas and are
nested from outermost to innermost.
Rule
Take each color's bbox as its completed rectangle, except that a child bbox
which already matches at least three sides of its parent's one-pixel inset
expands to that full inset; repaint the rectangles largest to smallest.
Key insight
The scattered pixels are samples of nested rectangular regions, so the
output is a clean redraw of those regions with inner colors overwriting
outer colors.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="42f83767" lines="23-39">
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

<reference_puzzle index="18" source="train" puzzle_id="7e2bad24" lines="91-106">
<description>
Extend the diagonal blue ray through the grid, reflecting it off red line-segment mirrors.

Input structure
- Black-background grid with one diagonal 8-connected blue ray seed.
- Red foreground objects, when present, are straight horizontal or vertical mirror segments.
- Other foreground bars are transparent to the ray and may be overwritten where the ray crosses them.

Rule
- Start from the ray seed endpoint whose outward diagonal extension remains inside the grid.
- Paint the ray continuously in that diagonal direction until it leaves the grid.
- If the next cell is a red mirror, do not paint it; reflect by flipping the row direction on a horizontal mirror or the column direction on a vertical mirror.

Key insight
- The non-red bars are decoys: only red bars reflect the diagonal path, while all other crossed cells become blue.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="e45ef808" lines="31-46">
<description>
Mark the shallowest and deepest columns of a solid terrain boundary with vertical colored guide lines.

Input structure:
- The first row is a solid empty separator.
- Below it, each column consists of upper fill cells followed by a contiguous suffix of terrain cells.
- The bottom row is all terrain, so every column has a well-defined first terrain row.

Rule:
- Find the column whose first terrain row is latest; it is unique and its upper fill run becomes maroon.
- Find the rightmost column whose first terrain row is earliest; its upper fill run becomes yellow.
- Preserve the empty separator, terrain cells, and all other upper fill cells.

Key insight:
- The task is a column-height summary of the terrain profile: color the deepest upper run and the highest terrain peak.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="271d71e2" lines="9-22">
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
