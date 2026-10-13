你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="df9fd884" lines="33-47">
<description>
Move the colored object from the lower bracket lane to the upper bracket lane and drop it to the bottom edge.

Input structure
- The background is the dominant color.
- One frame color forms exactly two 4-connected bracket objects: a top-edge bracket open downward and a bottom-edge bracket open upward.
- One other 4-connected foreground object sits in the lower bracket's interior column span.

Rule
- Remove the movable object from its original cells.
- Keep its shape and color unchanged.
- Paste it with its bbox columns aligned to the top bracket's interior columns and with its bbox bottom aligned to the grid bottom.

Key insight
- The lower bracket identifies the object's source lane; the upper bracket provides the target lane, and the final vertical placement is bottom-aligned.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="5587a8d0" lines="11-27">
<description>
Redraw foreground objects as nested square rings ordered by object size.

Input structure:
- The grid has one dominant background color.
- Every foreground object is a 4-connected monochromatic component.
- Foreground object areas are distinct.

Rule:
- Sort the foreground objects by decreasing pixel count.
- Create a square output with side length 2 * number_of_objects - 1.
- The largest object color fills the outer ring, each next object color fills
  the next inner ring, and the smallest object color fills the center pixel.

Key insight: The objects' original shapes do not get copied; their areas
determine only the outside-to-inside order of the nested square pattern.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="b0039139" lines="29-44">
<description>
Recolor a cropped template and repeat it according to a component-count stencil.

Input structure:
- Solid blue (1) separator rows or columns split the grid into four sections.
- Section 1 is a black-background shape template; section 2 is a component-count stencil.
- Sections 3 and 4 are solid color swatches for output foreground and background.

Rule:
- Crop section 1 to the bbox of its non-black pixels and recolor it with the two swatches.
- Count 4-connected non-black components in section 2.
- Repeat the recolored crop that many times, inserting one background-colored row/column between copies.

Key insight: The second section encodes only the repeat count; the output orientation
is inherited from whether the input separators are columns or rows.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="c9680e90" lines="52-67">
<description>
Move each mirrored bottom source pixel to the far end of its trail and redraw the top half as its reflection.

Input structure
- A solid non-background row separates equal-height top and bottom halves.
- The top half has one marker color on the background.
- In the bottom half, exactly one color has positions that mirror the top markers across the separator, and the other color forms trails touching those source pixels.

Rule
- Identify the bottom source color by reflecting each bottom color into the top half and matching the top marker set.
- For every bottom source pixel, follow its unique 4-connected trail of the other bottom color to the trail endpoint.
- Clear the old top markers, bottom sources, and trails; place the source color at each endpoint and place the top marker color at the reflected endpoint.

Key insight
- The top half is only a mirror display of the bottom moving sources; the actual motion is encoded by the trail attached to each bottom source pixel.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="18447a8d" lines="33-48">
<description>
Move each right-side colored shape into the left slot whose template color is its exact complement.

Input structure:
- The grid has two equal-width slots separated by one background column.
- Non-background rows form horizontal bands separated by all-background rows.
- Each band has a shared template color in the left slot and one colored source shape in the right slot.

Rule:
- For each band, define its destination shape as the cells in the left slot not occupied by the template color.
- Normalize every right-side source shape to right-slot coordinates and match it to the unique equal destination shape.
- Clear the original right-side source pixels and paint the matched left-slot complement with the source color.

Key insight:
The right-side objects are not placed by row order; each one completes the unique left template whose missing cells have the same shape.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="aaecdb9a" lines="10-26">
<description>
Convert scattered foreground objects into bottom-aligned count bars in a fixed five-color order.

Input structure
The grid has a dominant background color and foreground objects in the task
palette: grey, red, teal, maroon, and pink. Objects are monochromatic
8-connected components; some palette colors may be absent.

Rule
For each palette color in order grey, red, teal, maroon, pink, count its
8-connected objects. Create a background grid with five columns and height
equal to the largest count, then fill each column upward from the bottom
with its palette color for that count.

Key insight
The output does not preserve object shapes or positions; it is a histogram
of per-color 8-connected object counts, using the learned palette order.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="396d80d7" lines="18-30">
<description>
Paint the inner foreground color into every background pixel that touches the enclosing frame only diagonally.

Input structure:
- The grid has one dominant background color and exactly two foreground colors.
- One foreground color is an enclosing frame: its bbox strictly contains the bbox of the other, inner color.

Rule:
- Keep all existing pixels unchanged.
- For each background pixel, if it is diagonally adjacent to the frame color and has no orthogonal adjacency to the frame color, recolor it to the inner color.

Key insight: The inner color marks the color to use, while the frame's diagonal-only background neighbors define the cells to add.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="67e490f4" lines="8-22">
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

<reference_puzzle index="9" source="eval" puzzle_id="5dbc8537" lines="10-23">
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

<reference_puzzle index="10" source="train" puzzle_id="68bc2e87" lines="46-60">
<description>
Recover the bottom-to-top stacking order of overlaid rectangular color frames.

Input structure:
- The background is one color, and every foreground color is a one-cell-thick rectangular frame.
- Overlapping frame pixels show only the color of the frame that is highest at that location.

Rule:
- Reconstruct each frame as the border of the bounding box of its visible pixels.
- At every cell covered by multiple reconstructed frames, the visible frame color is above the other frames covering that cell.
- Return the unique bottom-to-top order of frame colors as a single-column grid.

Key insight:
The hidden parts of each frame do not need to be seen directly; the visible pixels reveal the full frame bbox, and overlap cells reveal the layer constraints.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="b74ca5d1" lines="157-172">
<description>
Swap each 5x5 sprite with its corner marker color and stamp same-marker composite masks into the matching corner.

Input structure:
- A 30x30 grid has one dominant background color.
- Non-background corner pixels are marker labels.
- Every other foreground color is the body of one 5x5 sprite containing exactly one pixel of a corner marker color.

Rule:
- For each sprite, recolor body pixels to the sprite marker color and recolor the marker pixel back to its body color.
- If a marker lies on the side facing its matching corner and another same-marker sprite has a body pixel at that same local cell, use that visible body color for the marker pixel.
- For each corner marker, overlay the masks of all sprites with that marker color into the corner 5x5 block and paint the visible composite with the marker color.

Key insight:
The corner pixels are labels that group 5x5 sprites by their embedded marker color; each labeled corner shows the grouped sprite silhouette.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="bd283c4a" lines="7-23">
<description>
Pack all pixels into color-frequency runs along bottom-up columns.

Input structure
The input is a rectangular grid whose colors are all foreground material;
only the number of pixels of each color matters. In the observed task
instances, every present color has a distinct pixel count.

Rule
Count the pixels of each color and order the colors from most frequent to
least frequent. Rewrite a same-size grid by placing each color's full count
as one contiguous run while scanning columns from left to right and, within
each column, from bottom to top.

Key insight
The output preserves the multiset of colors but discards all original
positions, turning the grid into a frequency-sorted column packing.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="a644e277" lines="31-50">
<description>
Extract the lattice subregion bounded by the four background-colored corner holes.

Input structure:
- A regular one-cell-thick lattice has one line color, one background color,
  and sparse marker pixels.
- Exactly four lattice-line intersections are background-colored holes instead
  of line-color pixels.

Rule:
- Find the unique non-background color that forms regular horizontal and
  vertical lattice lines.
- Use the four background-colored lattice intersections as the corners of one
  rectangle.
- Return the inclusive crop between those corner rows and columns.

Key insight:
- The crop is marked by the missing lattice intersections, not by the marker
  pixels inside the cells.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="ad3b40cf" lines="72-87">
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

<reference_puzzle index="15" source="eval" puzzle_id="dd6b8c4b" lines="9-22">
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

<reference_puzzle index="16" source="train" puzzle_id="5792cb4d" lines="44-59">
<description>
Reverse the sequence of colors along the single 4-connected foreground path while keeping the path shape fixed.

Input structure:
- The background is the dominant color.
- All non-background pixels form one multicolor 4-connected one-pixel-wide path.
- The path has exactly two endpoints and no branches.

Rule:
- Walk the path from one endpoint to the other.
- Replace the colors on those same path cells by the original path colors in reverse order.
- Leave every background pixel unchanged.

Key insight:
- Nothing moves geometrically; only the color string written on the path is reversed between the endpoints.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="9b30e358" lines="9-23">
<description>
Repeat the bottom foreground motif upward until it fills the whole grid.

Input structure
- The background is the dominant color.
- The only foreground pixels occupy a contiguous block of rows at the bottom.
- That bottom block is the complete vertical motif, including its background cells.

Rule
- Copy the bottom motif rows repeatedly upward across the full grid.
- Preserve horizontal alignment and vertical phase: the original first motif row remains at its input row.
- The output shape is unchanged.

Key insight
The blank rows above the bottom foreground block are missing repetitions of the same row motif, not empty space.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="b1986d4b" lines="19-37">
<description>
Render the filled-square object counts as compact nested stacks ordered by count rank.

Input structure
A large noisy grid has a dominant background, scattered non-square noise, and
filled monochrome square objects. Each square side length has exactly one
object color, and side lengths form consecutive sizes.

Rule
Ignore non-square components. If the lowest square in the scene is of the
smallest side length, treat that one smallest square as a distractor and do
not count it. For count rank k from left to right, create one block from all
square sizes with at least k remaining objects; draw larger squares first and
smaller squares over their upper-left corner, then append one background
separator column.

Key insight
The output is a count histogram of square side lengths, but each histogram
slot is rendered as nested upper-left square masks using the side colors.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="f0f8a26d" lines="30-44">
<description>
Rotate every foreground bar 90 degrees around its own center on the same canvas.

Input structure
The background is the dominant color, and all foreground pixels share one color.
Each 4-connected foreground object is a solid one-cell-thick horizontal or vertical
bar with odd length, with single-pixel bars allowed.

Rule
Clear the canvas to the background, then redraw every bar with its midpoint, length,
and color preserved while swapping horizontal and vertical orientation.

Key insight
The bars are independent objects: their centers and lengths identify the output
placement, so no matching, ordering, or global symmetry is needed.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="ecb67b6d" lines="138-155">
<description>
Recolor the grey diagonal-chain regions in an orange field to teal.

Input structure
The input contains only grey pixels and orange pixels.  Grey pixels form
several 8-connected regions embedded in the orange field.

Rule
Decompose each grey 8-connected region into 4-connected lobes.  Recolor the
whole region teal only when it contains a straight diagonal run of at least
three pixels and those lobes meet only by diagonal corner contacts in a
non-branching diagonal chain or cycle; a single 4-connected region is
recolored only when it is a square diagonal staircase.

Key insight
The teal regions are not selected by size or position; they are the grey
regions whose internal connectivity depends on diagonal corner joins, while
same-edge cracks, rectangles, bars, and branching/noisy regions remain grey.
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
