你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="9aaea919" lines="38-51">
<description>
Transfer sprites from red-marked lanes to green-marked lanes, recoloring and stacking.

Input structure:
- bg = most common color. Red (2) and green (3) thin bars define vertical lanes.
- Each lane contains repeated identical monochrome sprites in an evenly spaced vertical grid.

Rule:
- Each red lane's sprites turn grey (5); their count transfers to the nearest green lane.
- New sprites fill empty lattice slots bottom-up, using the source shape recolored to the destination lane's sprite color.
- All marker bars are removed.

Key insight: Sprite count is conserved across lanes — red sprites become grey, and equally many new sprites stack into the green lane's regular grid.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="d255d7a7" lines="99-114">
<description>
Move the patterned edge band to the opposite edge along every row or column whose band slice contains the trail color.

Input structure
- The dominant color is the background.
- The two foreground colors are a frequent trail color and a rarer marker color.
- All marker pixels lie nearest to one grid edge, and that edge contains trail-color runs that define the source-band thickness.

Rule
- Choose the source edge identified by the marker pixels.
- A row or column perpendicular to that edge moves if its source-band slice contains the trail color.
- Copy that slice unchanged to the opposite edge, erase the remaining trail cells to background, and keep marker-only lines fixed.

Key insight
- The marker color identifies the active edge, while the trail color identifies which parallel rows or columns carry an edge-band slice across the grid.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="b2bc3ffd" lines="8-23">
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

<reference_puzzle index="4" source="train" puzzle_id="42f83767" lines="23-39">
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

<reference_puzzle index="5" source="train" puzzle_id="ff2825db" lines="18-36">
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

<reference_puzzle index="6" source="train" puzzle_id="42f14c03" lines="75-83">
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

<reference_puzzle index="7" source="eval" puzzle_id="dfadab01" lines="79-93">
<description>
Expand each colored control pixel into its corresponding fixed 4x4 glyph.

Input structure:
- Black background with single-pixel markers in colors 2, 3, 5, and/or 8.
- Some inputs also show a sample 4x4 glyph in the output color, with a same-rule marker just below-right of the sample.

Rule:
- Replace each marker color by its learned 4x4 glyph: 2->yellow frame, 3->blue ring, 5->pink diagonal blocks, 8->orange X.
- Paint each glyph with its top-left corner at the marker pixel, clipping at the grid boundary.
- Remove the sample glyphs and skip their below-right label markers.

Key insight:
- The colored pixels are control pixels for a fixed glyph alphabet; non-control sample glyphs are only legend entries, not preserved scene objects.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="9720b24f" lines="34-46">
<description>
Delete each foreign-color object whose bounding box is contained by a larger object of another color.

Input structure
- The black pixels are the empty canvas, and every other color forms one or more 8-connected monochromatic objects.
- Some larger foreground objects act as loose U/C-like containers around smaller objects of different colors.
Rule
- Find all 8-connected monochromatic non-black objects.
- Recolor an object to black exactly when its bounding box is contained in the bounding box of a larger object of another color.
- Keep all other pixels unchanged.
Key insight
- The removable objects are identified by containment in another color's rectangular extent, not by their own color or shape.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="5ecac7f7" lines="23-36">
<description>
Combine the left, middle, and right ordered objects from three separated panels.

Input structure
- The grid is three HxH panels separated by one-cell vertical separator bars.
- The background is the dominant color, and each panel contains exactly three same-color 4-connected foreground objects.

Rule
- Build a new HxH background canvas.
- From the left panel copy its leftmost object, from the middle panel copy its middle object, and from the right panel copy its rightmost object.

Key insight
- The separator bars create three aligned layers, and the panel's own left-to-right object order selects which object survives in the merged output.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="dce56571" lines="8-23">
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

<reference_puzzle index="11" source="train" puzzle_id="edcc2ff0" lines="47-63">
<description>
Keep only the lower-panel colors named by the top markers and expand each marker into an object-count bar.

Input structure
A black instruction strip sits above a non-black rectangular scene panel.  The strip contains one-cell markers
in the first column, and each marker color is a candidate object color in the lower panel.  The lower panel has
a unique dominant background color and scattered 4-connected foreground objects.

Rule
In the lower panel, preserve objects whose colors appear as top markers and recolor every other foreground
object to the panel background.  In the instruction strip, replace each marker by a horizontal run from column 0
whose length is the number of 4-connected lower-panel objects of that marker color; erase markers with count 0.

Key insight
The top strip is a color filter plus a compact count summary: marker colors select which lower-panel objects
survive, and their bars report the number of surviving objects by color.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="2a28add5" lines="8-22">
<description>
Paint each pink control pixel as a teal horizontal bar whose reach equals the foreground counts on that row.

Input structure
- The grid has a dominant background color and one or more pink control pixels.
- Each marker row has exactly one pink pixel; all other non-background pixels
  in that row are counters.
Rule
- Start from an all-background output grid.
- For each pink pixel, count non-background pixels strictly to its left and
  strictly to its right in the same row.
- Paint teal from c - left_count through c + right_count on that row.
Key insight
- The foreground pixels are not copied; only their counts on each side of
  each pink marker encode the teal bar length.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="9f8de559" lines="19-37">
<description>
Erase the first border pixel hit by extending the red marker away from its pink trail.

Input structure
- The dominant background surrounds a unique red marker and a four-pixel
  pink trail.
- The pink pixels form one straight 8-connected ray starting next to the
  red marker.
- In the opposite direction from that trail, the red marker faces a
  non-background border or obstacle pixel beyond background cells.

Rule
- Keep the whole grid fixed except for the first non-background pixel on
  the ray from the red marker away from the pink trail.
- Recolor that one hit pixel to the background color.

Key insight
- The pink trail marks where the red marker came from, so the edit is found
  by ray-casting from the red marker in the opposite direction.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="5a719d11" lines="44-59">
<description>
Within each horizontal pair of two-color panels, swap the motif masks and recolor each pasted mask with the source panel's base color.

Input structure:
- A separator color forms full blank rows and at least one full blank column.
- The non-separator area is split into row bands, each containing exactly two same-shaped rectangular panels.
- Each panel has a dominant base color and one minority motif color.

Rule:
- For every row band, clear each panel to its own base color.
- Paste the right panel's motif mask into the left panel using the right base color.
- Paste the left panel's motif mask into the right panel using the left base color.

Key insight:
- The motif colors themselves are distractors; only each motif's shape mask and the two panel base colors are exchanged.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="7666fa5d" lines="12-26">
<description>
Fill the parallelogram-shaped region between parallel anti-diagonal line segments with red.

Input structure:
- bg = most common color.
- One non-bg color forms short diagonal segments, each lying on a single anti-diagonal (r+c = const).
- Segments come in parallel pairs bounding a region to be filled.

Rule:
- For each main-diagonal axis (r−c = const), collect anti-diagonal segments whose span covers that axis.
- Sort those segments by their anti-diagonal index (r+c).
- Fill all bg cells lying strictly between consecutive segment pairs along that axis with red (color 2).

Key insight: The two diagonal directions decouple — segments sit on anti-diagonals, and filling sweeps along main diagonals between them, forming parallelogram regions.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="6e4f6532" lines="26-41">
<description>
Dock train sprites onto station groups by matching coupler count, then orient via divider-color markers.

Input structure:
- bg; colored stripes (edge + internal) partition grid into rectangular regions.
- Color 8 forms train bodies; color 9 marks both isolated stations and train-attached couplers.
- Stripe-colored pixels on trains are orientation markers.

Rule:
- Stations = 8-connected 9-clusters not adjacent to any 8. Trains = components containing 8.
- Match train to station group where |couplers| == |station size|.
- Rotate/flip train so marker pixels face the same-color boundary stripe of the target region.
- Translate so couplers align exactly onto station cells; erase original position.

Key insight: Coupler count selects the docking target; marker color encodes which boundary stripe to face, fixing the unique orientation.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="5dbc8537" lines="10-23">
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

<reference_puzzle index="18" source="train" puzzle_id="f0100645" lines="51-68">
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

<reference_puzzle index="19" source="train" puzzle_id="5751f35e" lines="29-43">
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

<reference_puzzle index="20" source="train" puzzle_id="25e02866" lines="14-31">
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
