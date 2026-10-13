你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="98c475bf" lines="62-76">
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

<reference_puzzle index="2" source="train" puzzle_id="2e65ae53" lines="88-103">
<description>
Fill every lattice slot from the color assigned to that slot position by the other lattices.

Input structure:
- The background is the dominant color.
- The largest foreground color forms several one-pixel-thick rectangular lattice frames.
- The frame lines divide each lattice into the same row-by-column set of rectangular slots; some slots already contain one non-frame color.

Rule:
- Infer a color for each logical slot position from the colored slots across all lattice frames.
- Repaint every slot interior in every lattice with the inferred color for its row/column position.
- Keep the frame lines and all background outside frames unchanged.

Key insight:
- Slot sizes vary by frame, but the complete color pattern is indexed by lattice slot position rather than by absolute pixel size or location.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="4e7e0eb9" lines="31-49">
<description>
Transform each 9x9 panel by mirroring across its center guide line or filling all four block slots from its lone exemplar.

Input structure
The grid is made of 9x9 panels separated by one-cell-wide grid lines. Each
panel has four monochrome 3x3 slots at the quadrant positions, with either a
full non-background center row, a full non-background center column, or no
guide line.

Rule
A panel with a full center column is reflected left-to-right, and a panel
with a full center row is reflected top-to-bottom. A panel with no guide line
has exactly one slot color that appears once and one slot color that appears
three times; recolor all four slots to the once-occurring slot color.

Key insight
The center guide line encodes the mirror axis, while panels without a guide
line are completed by copying the structurally unique slot color.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="1d61978c" lines="8-22">
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

<reference_puzzle index="5" source="eval" puzzle_id="b10624e5" lines="24-39">
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

<reference_puzzle index="6" source="train" puzzle_id="42f83767" lines="23-39">
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

<reference_puzzle index="7" source="eval" puzzle_id="a395ee82" lines="9-24">
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

<reference_puzzle index="8" source="train" puzzle_id="84551f4c" lines="42-53">
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

<reference_puzzle index="9" source="train" puzzle_id="ad173014" lines="97-111">
<description>
Recolor each framed interior mask with the color from the next red frame clockwise around the blue connector cycle.

Input structure
- The grid contains hollow red square frames connected pairwise by blue line components.
- The blue components connect the frames into one cycle, with every frame touching exactly two blue components.
- Each frame interior contains background plus exactly one non-background color mask.

Rule
- Sort the red frames clockwise by their centers around the cycle.
- Keep each frame's interior mask shape in place, but recolor it with the interior color from the next clockwise frame.
- Leave the red frames, blue connectors, background, and interior background holes unchanged.

Key insight
- The colored masks do not move; the blue cycle gives the order in which their colors rotate.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="af726779" lines="17-35">
<description>
Grow an alternating pink/orange triangular trace downward from the horizontal orange seed row.

Input structure:
- The grid has a green background and one horizontal row containing all orange foreground pixels.
- The seed row may contain isolated orange pixels and adjacent orange runs.
- Pink is absent from the input and is used for the first generated row.

Rule:
- Copy the input grid.
- Two rows below the current active row, place a child pixel between each active pair
  whose middle cell in the current row is background.
- Alternate generated row colors between pink and orange, stopping when no children
  can be generated or the next row would leave the grid.

Key insight:
- Only an active-background-active triplet in the current row reproduces; contiguous
  active runs do not produce children through occupied middle cells.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="5dbc8537" lines="10-23">
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

<reference_puzzle index="12" source="train" puzzle_id="dce56571" lines="8-23">
<description>
Straighten the single foreground mask into a centered horizontal segment of the same color and length.

Input structure:
- The grid has one dominant background color.
- All non-background pixels share one foreground color.
- The number of foreground pixels fits in one centered row of the grid.

Rule:
- Erase the original foreground mask.
- Count its pixels, keep its color, and paint that many pixels as a horizontal segment.
- Place the segment on the middle row with equal background margin on the left and right.

Key insight:
- The original foreground shape is irrelevant except for its color and area; the output is the area normalized into the grid's central horizontal line.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="6ffe8f07" lines="49-64">
<description>
Cast yellow horizontal and vertical rays from the teal rectangle until blue blockers stop them.

Input structure:
- A black-background grid containing one solid teal rectangle.
- Blue objects are blockers; other non-background objects may lie in the ray paths.
- Yellow is absent from the input and is the output paint color.

Rule:
- From every row of the teal rectangle, cast rays left and right to the grid edge.
- From every column of the teal rectangle, cast rays up and down to the grid edge.
- A ray stops before a blue pixel; all crossed non-blue, non-teal pixels become yellow.

Key insight:
The teal rectangle defines a full cross-shaped projection, while blue objects are walls and red/background cells are simply paintable space.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="c3fa4749" lines="79-96">
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

<reference_puzzle index="15" source="train" puzzle_id="6ca952ad" lines="15-30">
<description>
Drop every large foreground object straight down to the bottom while leaving small marker objects fixed.

Input structure:
- The grid has a dominant background color and exactly one foreground color.
- The foreground is split into separated 4-connected objects.
- Objects of area at most three are stationary markers; larger objects are moving sprites.

Rule:
- Clear each moving sprite from its original position.
- Translate each moving sprite vertically downward until its bbox bottom reaches the grid bottom.
- Repaint each translated sprite in its original color; small markers stay unchanged.

Key insight:
- Object area separates the active falling sprites from the small fixed markers, and all active sprites follow the same vertical path.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="80214e03" lines="48-64">
<description>
Extract the color of each noisy block in a row/column lattice and mirror the coarse grid horizontally.

Input structure:
- A background surrounds a rectangular lattice of monochrome foreground blocks.
- Consecutive rows in a lattice band share the same set of visible foreground colors.
- Consecutive columns in a lattice band share the same set of visible foreground colors.
- Each row-band/column-band intersection has one foreground color, with possible background holes.

Rule:
- Group non-empty rows and columns into consecutive bands by their foreground color sets.
- Replace each lattice cell by the single foreground color inside that row/column-band intersection.
- Return the resulting coarse color grid with the column order reversed.

Key insight:
- Whole-row and whole-column color sets reveal the hidden lattice even when neighboring cells share a color or contain holes.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="faa9f03d" lines="36-52">
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

<reference_puzzle index="18" source="eval" puzzle_id="0934a4d8" lines="75-90">
<description>
Reveal the rectangular patch hidden by the solid marker using the grid's reflection symmetries.

Input structure:
- A square colored pattern with one solid rectangular marker color occluding a patch.
- The visible pattern has horizontal and vertical reflection symmetry on a large lower/right strip.
- The top and left outer strips also expose transpose counterparts for cells not recoverable by those reflections.

Rule:
- Find the unique color whose full extent is a solid rectangle; this rectangle is the output region.
- Infer the reflected strip start as the smallest row/column offset whose lower/right strips are mirror-consistent, ignoring marker pixels.
- For each marker pixel, copy the unique non-marker value in its horizontal/vertical reflection orbit; if that orbit is entirely hidden in the outer strip, use the transposed top/left-strip counterpart.

Key insight:
The marker is not part of the pattern; it covers a missing rectangle whose cells are still determined by the visible mirror or transpose copies elsewhere in the grid.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="ff2825db" lines="18-36">
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

<reference_puzzle index="20" source="train" puzzle_id="3cd86f4f" lines="6-18">
<description>
Shear the rectangular input down-left onto a black canvas.

Input structure
A rectangular grid of colored pixels with no internal empty region to infer.

Rule
Create a black output with the same height and width W + H - 1, then copy
each input row r unchanged starting at column H - 1 - r.

Key insight
The colors and row contents are preserved; only the row offsets change so
the left edge forms a descending diagonal.
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

请一次性生成5个`<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
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

