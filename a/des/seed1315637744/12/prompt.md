你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="dd6b8c4b" lines="9-22">
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

<reference_puzzle index="2" source="eval" puzzle_id="58490d8a" lines="9-21">
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

<reference_puzzle index="3" source="train" puzzle_id="5b37cb25" lines="59-73">
<description>
Fill each plus-shaped background notch with the color of the frame side it opens toward.

Input structure
- A one-cell-thick rectangular frame has four solid-colored sides and neutral corner pixels.
- The frame interior contains a dominant background color and one foreground object color.
- Some object boundaries contain five-pixel plus-shaped background notches.

Rule
- Find every background plus whose opposite side is supported by foreground object pixels and whose open side faces one frame side.
- Recolor the five pixels of that plus with the color of the facing frame side.
- Leave all other pixels unchanged.

Key insight
The frame side colors are direction labels for the open notches in the foreground object.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="f8cc533f" lines="10-25">
<description>
Complete every colored sprite by applying the union of all aligned partial sprite masks.

Input structure:
- The dominant color is the background.
- Each non-background color appears as a separated partial copy of one common sprite mask.
- The copies share a top-left-aligned sprite window; some copies may be missing lower or right edge pixels.

Rule:
- Crop each same-color mask to its bounding box.
- Pad the crops to the maximum observed height and width, aligned at the top-left, and OR them into one complete template.
- Fill that template at every sprite location using the sprite's own color.

Key insight:
- No single input sprite is complete, but their missing cells are complementary, so the aligned union reveals the full mask.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="a32d8b75" lines="262-275">
<description>
Assemble and place enlarged sprites on a canvas using legend-panel instructions.

Input structure:
- Color-6 separators divide the grid into a left legend column and a main canvas.
- Each legend panel has (top to bottom): a unit sprite, a frame template, a 5x5 bordered square with a position-marker pixel, and a 3x3 rotation clock-hand.

Rule:
- Tile the unit sprite into each active cell of the frame to build an enlarged composite.
- Rotate the composite per the clock-hand direction (up=0, right=90, down=180, left=270 degrees).
- Place it on the canvas anchored at the position encoded by the marker pixel's row/col in the bordered square's 3x3 inner grid.

Key insight: The marker pixel's position in a 3x3 grid jointly encodes vertical and horizontal anchor alignment (start/middle/end).
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="dbff022c" lines="275-288">
<description>
Fill hollow sprites' interiors using a color-palette legend on the grid edge.

Input structure:
- bg = most common color. Several hollow sprites with distinct border colors.
- A small 2×N or N×2 palette block sits on one grid edge; one row/column lists border colors, the other lists corresponding fill colors.

Rule:
- Locate the palette legend on a grid edge and read it as a border→fill color mapping.
- Find enclosed bg regions (holes) inside sprites that don't touch the grid boundary.
- For each hole, identify the most common adjacent border color, look it up in the palette, and fill the hole with the mapped fill color.

Key insight: The palette's edge position and orientation determine reading direction; border colors in the palette uniquely key into fill colors.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="52df9849" lines="49-67">
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

<reference_puzzle index="8" source="eval" puzzle_id="7666fa5d" lines="12-26">
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

<reference_puzzle index="9" source="eval" puzzle_id="0934a4d8" lines="75-90">
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

<reference_puzzle index="10" source="train" puzzle_id="a6953f00" lines="45-63">
<description>
Extract the top-left or top-right 2x2 corner crop selected by the top edge clue.

Input structure
- The input is a square color grid whose answer is one of the two 2x2 crops
  touching the top edge and a grid corner.
- A repeated color on the top row, when present, acts as the primary side cue.
- Otherwise, the two top corner colors have different whole-grid frequencies.

Rule
- If a top-row color repeats, find the repeated top-row occurrence nearest to
  a left or right edge and take the 2x2 top-corner crop on that side.
- If the top row has no repeated color, compare the whole-grid frequencies of
  the two top-corner colors and take the side whose corner color is rarer.

Key insight
- The task is not asking for an arbitrary 2x2 search: the decisive information
  is on the top edge, and it selects which top corner crop to keep.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="6cbe9eb8" lines="79-94">
<description>
Rebuild the foreground rectangles as a largest-to-smallest lower-left nesting.

Input structure
- A periodic wallpaper background is overlaid by foreground colors that each form a rectangular frame or solid rectangle.
- The largest foreground rectangle is the output frame; smaller foreground rectangles are the objects to nest inside it.
- A frame may be partially hidden by a smaller rectangle, but its visible pixels still determine its bbox.

Rule
- Ignore the wallpaper and crop the result to the largest foreground rectangle's bbox size.
- Sort foreground rectangles by bbox area from largest to smallest.
- Reconstruct each frame or solid rectangle, placing every next smaller rectangle in the lower-left of the current rectangle's usable region.

Key insight
- The scattered background colors are not objects; the meaningful order is the unique descending rectangle size chain, and each step nests into the previous rectangle's lower-left interior.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="7ed72f31" lines="8-22">
<description>
Mirror colored sprites using red (2) markers as reflection axes.

Input structure:
- bg = most common color. Red (2) = marker color.
- Colored sprites appear near red markers.

Rule:
- Red line within a sprite (vertical or horizontal) → mirror payload across that axis.
- Single red pixel in a mixed sprite → point-reflect payload around it.
- Isolated red pixel (alone, no payload) → point-reflect each adjacent 8-connected color component.
- Originals remain; mirrored pixels overlay.

Key insight: Red marker geometry (line vs single pixel) determines reflection type; isolated vs embedded red determines what gets reflected.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="e12f9a14" lines="138-152">
<description>
Emit diagonal beams from framed 2x2 blocks through frame gaps; beams interact on collision.

Input structure:
- bg = most common color. Wall color (e.g. green) forms 4x4 frames around 2x2 monochrome blocks.
- Each frame has one or more gap pixels where the wall is missing.

Rule:
- Each gap emits a beam of the inner block's color, traveling diagonally outward (direction set by gap position relative to the block).
- Beams paint every cell they cross until they hit the grid edge or collide.
- 90° collision: both beams turn 45° toward their bisector direction; the intersection cell is left blank (1-pixel gap).
- 45° collision: beams merge into a zigzag (alternating cardinal + diagonal steps) with no gap at the meeting point.

Key insight: Two distinct collision rules — 90° produces a gap and single-pixel diagonal lines; 45° produces no gap and thicker zigzag bands.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="758abdf0" lines="54-69">
<description>
Advance short marker bars away from the full edge bar, turning completed two-pixel markers into edge-color marks at the opposite side.

Input structure
- The background is the dominant color.
- One non-background color forms a complete one-cell-thick bar on exactly one outer edge.
- The other foreground color appears only in the first two cells inward from that bar, lane by lane.

Rule
- In each lane perpendicular to the edge bar, a one-pixel marker adjacent to the bar grows one more cell inward.
- A two-pixel marker in those first two inward cells is cleared back to background.
- The cleared two-pixel marker is replaced by a two-pixel mark of the edge-bar color at the opposite side of the same lane.

Key insight
- The full edge bar defines both the orientation and the direction of the local two-step marker cycle.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="df978a02" lines="57-72">
<description>
Let the largest center-pointing object grow outward while every other object loses its inward tip.

Input structure
- The grid has a single dominant background color and four 8-connected monochromatic foreground objects.
- The objects surround a central background gap and point toward it from cardinal directions.
- Exactly one object has the largest area.

Rule
- Find the central background gap by minimizing aggregate distance to the four objects.
- Recolor the nearest-to-center pixel of each non-largest object to background.
- Add a three-pixel cap in the largest object's color on the side away from the center, centered on the shared row or column.

Key insight
- The edit is a transfer of prominence around a common center: losers retract at their tips, and the unique largest object extends outward from the center.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="15660dd6" lines="33-54">
<description>
Combine each column's repeated template mask with its single solid color swatch.

Input structure:
- The grid is a table of framed panels separated by background rows/columns.
- The leftmost column gives one solid row-label color for each panel row.
- In each panel column, exactly one panel has a solid non-background interior,
  while the other panels share the same mask drawn in one template color on
  the background.

Rule:
- For every panel column, find the row containing the solid interior swatch.
- Rebuild one output panel using that row's left label as the frame color.
- Fill the repeated template mask with the swatch color and leave the mask
  background cells as background; separate rebuilt panels with one
  background column.

Key insight:
The table encodes a colorized stencil per column: the solid panel supplies
the paint color and frame label, while the duplicate template panels supply
the shape mask.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="abbfd121" lines="56-74">
<description>
Recover the repeating background pattern hidden under the largest solid rectangle.

Input structure:
- The grid is covered by a small row/column periodic pattern.
- Several non-pattern colors form solid rectangular foreground objects that
  overwrite the pattern.
- The requested output region is the largest of those solid rectangles.

Rule:
- Find every 4-connected monochromatic object whose bounding box is a filled
  rectangle with both dimensions greater than one.
- Treat those rectangles as masks, infer the smallest repeating tile that is
  consistent with all remaining visible pixels, and repaint the largest
  rectangle's coordinates with that tile.

Key insight: The rectangle colors are occluders; the visible pixels outside
them are enough to reconstruct the original periodic background underneath.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="18447a8d" lines="33-48">
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

<reference_puzzle index="19" source="train" puzzle_id="af726779" lines="17-35">
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

<reference_puzzle index="20" source="eval" puzzle_id="58f5dbd5" lines="20-34">
<description>
Carve color-matched motifs into solid blocks and pack them into a compact grid.

Input structure:
- bg = most common color.
- Several solid 5×5 monochrome blocks, each a distinct non-bg color.
- For each block color, a small same-colored motif sprite elsewhere on the grid.

Rule:
- Match each motif to its same-colored 5×5 block.
- Center the motif within the block and recolor those cells to bg (punching holes).
- Pack all carved blocks into an output grid preserving their original row/column order, separated by 1-cell gaps and a 1-cell outer border.

Key insight: The motif acts as a stencil — it's not stamped onto the block but carved out of it as background holes.
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

