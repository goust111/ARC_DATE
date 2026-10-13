你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="4c7dc4dd" lines="7-21">
<description>
Learn a transformation rule from a training pair of framed sprites, then apply it to a test sprite.

Input structure:
- Grid contains several colored square frames (hollow borders of a single color).
- Each frame's interior holds a small pattern sprite; one frame's interior is blank (all black).
- Two frames form a training pair (their interiors share the same color palette); one remaining frame is the test case.

Rule:
- Identify the training pair by matching interior color sets; the other non-blank frame is the test input.
- Infer the rule mapping source→destination (e.g. reflection, color inversion, gap filling, endpoint extraction, or checkerboard complement).
- Apply that same rule to the test sprite; output the transformed interior.

Key insight: The puzzle is an analogy — "A is to B as C is to ?" — where you must discover the visual rule from one example pair and generalize it.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="ff2825db" lines="18-36">
<description>
Select the majority interior marker color, then redraw the framed board as that color's frame plus its bounding-box outline.

Input structure:
- The top row is a copied palette made from adjacent equal color pairs.
- Below it is a one-cell-thick rectangular frame around a black interior.
- Non-black, non-frame interior pixels are marker colors; one marker color has
  the unique highest count.

Rule:
- Keep the palette row unchanged.
- Pick the uniquely most frequent interior marker color.
- Clear the lower board to black, recolor its outer frame with the picked color,
  and draw the picked color's interior bounding-box border.

Key insight:
- The noisy marker cloud is not preserved as pixels; only the majority marker
  color and its spatial extent define the output.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="1d61978c" lines="8-22">
<description>
Recolor the two diagonal foreground-bar orientations, assigning teal to the larger orientation and red to the smaller one.

Input structure
- The grid has one dominant background color and one foreground color.
- Foreground pixels form one-cell-thick diagonal bars.
- Every foreground pixel belongs to exactly one diagonal orientation.

Rule
- Keep the background unchanged.
- Classify each foreground pixel by whether it has a diagonal neighbor on the NW/SE axis or the NE/SW axis.
- Recolor the orientation with more foreground pixels teal and the other orientation red.

Key insight
- The output colors depend on the relative population of the two diagonal orientations, not on a fixed slash direction or position.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="58c02a16" lines="10-26">
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

<reference_puzzle index="5" source="train" puzzle_id="9caf5b84" lines="28-45">
<description>
Recolor every pixel whose color is not one of the two most frequent input colors to orange.

Input structure:
- The grid contains at least three colors.
- Orange is absent from the input and is used only as the output replacement color.
- Exactly two colors occupy the top two frequency ranks; they may tie with each
  other, but no third color ties them at the cutoff.

Rule:
- Preserve every pixel whose color is one of the two most frequent input colors.
- Replace every other pixel with orange.

Key insight:
- The visible distinction is not the numeric identity of the colors but their
  population: the two majority colors remain, while all minority colors merge
  into the new orange color.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="97c75046" lines="88-107">
<description>
Move the single marker to the endpoint reached by sliding along the obstacle contour.

Input structure:
- A dominant background color fills the canvas.
- One single-pixel marker sits away from a connected obstacle region of the
  remaining color.
- The marker has exactly one horizontal or vertical sightline to the obstacle.

Rule:
- Find the last background pixel before the obstacle along that sightline.
- From that pixel, try the two diagonal turns that continue around the
  obstacle boundary.
- Move the marker to the endpoint of the turn that remains 4-adjacent to the
  obstacle for the longest distance.

Key insight:
The marker is not placed by distance to the obstacle; it first points to the
obstacle, then follows the visible outside contour until that contour segment ends.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="9f669b64" lines="26-48">
<description>
Push the central solid rectangle to the grid edge, splitting the perpendicular target rectangle around its path.

Input structure
- The background is the dominant color.
- There are exactly three monochrome foreground objects arranged on one
  horizontal or vertical axis.
- The middle object on that axis is a solid rectangular moving object.
- One end object is a solid rectangle or bar whose long side is perpendicular
  to the movement axis; this is the target to split.

Rule
- Move the middle rectangle along the axis toward the target until it is
  flush with the grid edge in that direction.
- Clear the original middle rectangle and the original target.
- Split the target through the moving rectangle's two-cell-wide path and
  shift the two halves one cell outward perpendicular to the movement.
- Keep the other end object unchanged.

Key insight
- The non-target end object acts as a pusher; the central rectangle travels
  away from it through the perpendicular target, opening a matching gap.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="faa9f03d" lines="36-52">
<description>
Repair a one-cell-thick colored line network by replacing singleton markers, filling short orthogonal gaps, and redirecting yellow-marked arms to the edge.

Input structure:
- A rectangular grid with a dominant background.
- Line colors form one-cell-thick horizontal/vertical stroke fragments.
- Marker colors are exactly those whose 4-connected components are all singleton pixels.
- Red singleton markers are missing line cells; yellow singleton markers are directional controls next to a line endpoint.

Rule:
- Assign every marker to the nearest visible line color in row/column line-of-sight, preferring an endpoint supported along that same axis.
- For each line color, fill the gap between consecutive facing endpoints in a row or column.
- A yellow control erases the old straight arm on the far side of its endpoint and paints the same color from that endpoint through the control pixel to the grid edge.
- At crossings, yellow-control rays win; otherwise an original-vs-filled crossing uses the more fragmented pre-repair line color, while pure filled crossings use the lower color id.

Key insight: The singleton markers identify which sparse stroke fragments belong together; after they are colored, the intended lines are the axis-aligned gaps between facing endpoints, with yellow controls flipping one arm to the marked side.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="5961cc34" lines="8-21">
<description>
Build a red ray network by activating blue blobs via directional green ports.

Input structure:
- bg = most common color. Large blue blobs with adjacent green pixels marking port directions.
- Start marker: yellow pixel atop a short red segment at grid bottom.

Rule:
- Extend a ray upward from start; on hitting a blue blob, recolor it red.
- Activated blobs emit thick rays through green ports until boundary or another blob.
- Cascade until no new blobs are reached.

Key insight: Green pixels encode ray direction and width; the network activates as a chain reaction from one start point.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="fe45cba4" lines="72-89">
<description>
Move the left-edge fragment into its same-colored anchor fragment to complete a solid rectangle.

Input structure
- The grid has a dominant background color.
- Exactly one foreground color is split into two 4-connected objects: one
  touches the left edge, and the other is an anchor away from the left edge.
- The translated left-edge object and the anchor can form one filled rectangle.
Rule
- Find the unique split color whose left-edge object can be translated onto
  background cells beside the other same-colored object so their union is a
  solid axis-aligned rectangle.
- Clear the left-edge object's original cells to the background.
- Paint that object at the unique completing offset, preserving all other cells.
Key insight
- The active color is identified by object structure and rectangle completion,
  not by a fixed color id; the same-colored anchor object stays in place.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="9b5080bb" lines="161-172">
<description>
Recolor the small defect bars on each shared-color object with the opposite panel color.

Input structure
- The grid contains two large panel colors and one shared object color.
- Each shared-color object is a mostly solid rectangle inside exactly one panel.
Rule
- Find the near-solid rectangular core of each object.
- Missing core pixels and one-cell protrusions define two-pixel bars perpendicular to the nearest core edge.
- Recolor those bars with the other panel color; leave all other pixels unchanged.
Key insight
- The colors to add are not tied to numeric ids: each object's surrounding panel identifies the opposite panel color.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="c4d1a9ae" lines="28-48">
<description>
Fill each vertical foreground band's background cells with the color of the next band to its right.

Input structure
- The dominant color is the background.
- Each other color occupies a separate vertical band defined by that color's
  minimum and maximum columns.
- Within a band, all cells are either that band's color or background.

Rule
- Order the bands from left to right.
- In each band, replace background pixels with the color of the next band in
  that order, wrapping from the rightmost band back to the leftmost.
- Preserve all existing foreground pixels and all background pixels outside
  the bands.

Key insight
- The column bands form a cyclic palette legend: a band's gaps are completed
  by borrowing the color from its right-hand neighbor, not by changing the
  band's existing pixels.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="3bd292e8" lines="13-29">
<description>
Recolor wall-separated background regions alternately, starting with grey in the lower-left region.

Input structure
- The grid has one dominant background color and one foreground wall color.
- The wall pixels form one or more 4-connected one-cell-thick segments.
- Each wall segment separates exactly two 4-connected background regions.
Rule
- Keep all wall pixels unchanged.
- Label the 4-connected background regions and connect two labels when they
  touch opposite sides of the same wall segment.
- Color the region containing the lower-left corner grey, then flip between
  grey and green across every wall adjacency.
Key insight
- The red walls define a bipartite map of regions; the lower-left background
  region fixes which side of the alternating coloring is grey.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="5034a0b5" lines="16-32">
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

<reference_puzzle index="15" source="train" puzzle_id="83eb0a57" lines="72-86">
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

<reference_puzzle index="16" source="train" puzzle_id="6a980be1" lines="46-61">
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

<reference_puzzle index="17" source="train" puzzle_id="7e4d4f7c" lines="9-24">
<description>
Compress the L-shaped cue to three rows and add a pink copy of the top-row foreground mask.

Input structure
The grid has a dominant background color, a two-color top row, and all
foreground pixels below the top row lie in the first column as vertical markers.

Rule
Keep the input top row as output row 0 and the first marker row as output row 1.
Output row 2 is background except at the top-row foreground columns, which are
recolored pink.

Key insight
The lower repeated marker rows only identify the vertical cue; the generated
row is a recolored mask of the horizontal cue in the top row.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="45a5af55" lines="5-18">
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

<reference_puzzle index="19" source="train" puzzle_id="c803e39c" lines="31-40">
<description>
Expand the second mask into a block grid by stamping the first mask with the two payload colors.

Input structure
- A background grid has four panels separated by three full-height separator columns.
- The first two panels contain same-size binary masks; the last two contain solid same-size color squares.
Rule
- Use the second mask as an outer layout. For each foreground cell in that outer mask, place a tile equal to the first mask, coloring its foreground with the first solid-square color and its background with the second solid-square color; outer background cells become all second solid-square color.
Key insight
- The output is the Kronecker product of the two binary masks, with the two right-hand panel colors providing the palette.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="e45ef808" lines="31-46">
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
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
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

