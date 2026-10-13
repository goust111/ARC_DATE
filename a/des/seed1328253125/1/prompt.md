你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="31f7f899" lines="7-21">
<description>
Sort vertical bars by height and redistribute them across fixed colored slots.

Input structure:
- bg background; a horizontal base line of a single color spans the full grid width.
- Several monochrome vertical bars of distinct colors rise upward from the base line.

Rule:
- Rank bar shapes by height (shortest to tallest).
- Rank original x-positions (slots) left to right.
- Place the i-th shortest shape into the i-th leftmost slot, recoloring it to that slot's original color.
- The base line is preserved unchanged.

Key insight: Each slot keeps its x-position and color identity; only the bar silhouettes are redistributed among slots by height order.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="9bbf930d" lines="61-72">
<description>
Move magenta particles along corridors formed by parallel colored bars.

Input structure:
- bg=7. Particles=6 in col 0. Colored bars on even rows form corridors on odd rows.

Rule:
- Particle goes straight through bg cells.
- When blocked by a colored cell, turns 90° into the perpendicular direction
  that has a bg cell flanked by same-color bars. Stops at grid edge or full blockage.

Key insight: Same-color parallel bars define the only valid corridors, so each particle follows a deterministic local trace.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="13e47133" lines="56-69">
<description>
Fill partitioned regions with concentric colored bands derived from marker pixels.

Input structure:
- Grid partitioned into regions by wall-colored lines.
- Each region has a dominant base color and a few isolated marker pixels of other colors at varying depths from the region boundary.

Rule:
- For each region, read marker colors ordered by their Chebyshev distance from the boundary to build a color palette.
- If no marker sits on the boundary itself, the base color leads the palette.
- Paint every cell with palette[distance mod palette_length], producing repeating concentric rectangular frames that radiate inward.

Key insight: Each marker's depth from the boundary defines its position in the cyclic palette; the pattern then tiles inward automatically.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="e3721c99" lines="69-83">
<description>
Recolor grey shapes by matching their hole count to a color legend.

Input structure:
- bg = most common color; grey (5) sprites scattered across the grid.
- Colored legend shapes (non-bg, non-grey, at least 2×2, not touching the outer border) near grid edge,
  each with a distinct number of enclosed holes.

Rule:
- Each colored legend shape defines a mapping: hole count → color.
- For every grey sprite, count its holes (enclosed bg regions not touching the sprite's bounding-box border).
- Recolor the grey sprite to the matching legend color; erase grey distractors with unlisted hole counts.

Key insight: Shape identity is encoded by topology (number of holes), not geometry or size.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="221dfab4" lines="6-19">
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

<reference_puzzle index="6" source="eval" puzzle_id="89565ca0" lines="120-136">
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

<reference_puzzle index="7" source="eval" puzzle_id="eee78d87" lines="40-55">
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

<reference_puzzle index="8" source="eval" puzzle_id="a32d8b75" lines="262-275">
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

<reference_puzzle index="9" source="eval" puzzle_id="409aa875" lines="68-81">
<description>
Fire projectiles from 3-pixel triangle sprites; mark collisions with maroon or blue.

Input structure:
- bg = most common color. Non-bg 3-pixel sprites shaped as axis-aligned L's or diagonal V's.

Rule:
- Each triangle has an apex and an emission direction determined by its shape: L-shapes fire diagonally away from the missing corner; V-shapes fire along the axis away from the open side.
- A projectile lands exactly 5 cells from the apex in the emission direction.
- If it hits an existing sprite, that entire sprite is recolored to maroon (9).
- If two projectiles land on the same empty cell, that cell becomes blue (1); a lone projectile on empty bg paints maroon (9).

Key insight: The triangle's orientation encodes both direction and target; dual-projectile collisions on open bg produce blue, not maroon.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="dbff022c" lines="275-288">
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

<reference_puzzle index="11" source="eval" puzzle_id="e87109e9" lines="68-83">
<description>
Emit four teal beams from a square, turning at colored obstacles per a legend.

Input structure:
- Top band: grey (5) legend with colored vertical bars in cells; bar position (left or right) encodes turn direction.
- Below: main grid with bg, a solid teal (8) square, and colored obstacle blocks.

Rule:
- Remove the legend from the output.
- From the teal square, shoot a beam in each cardinal direction (as a sliding copy of the square).
- Each beam paints teal over bg cells as it moves.
- On hitting an obstacle of color C, consult the legend: bar on left → turn counterclockwise, bar on right → turn clockwise. Continue in the new direction.
- Beams stop at grid edges.

Key insight: The legend's bar position (left vs right within its cell) is the encoding for turn direction at each obstacle color.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="2d0172a1" lines="209-224">
<description>
Convert a rough nested line drawing into a clean canonical frame-and-dot diagram.

Input structure:
- The grid has one background color and one foreground color.
- Foreground 4-connected objects are rough closed frames or small filled blobs.
- Frames may contain other frames/blobs; blobs outside a frame remain outside it.

Rule:
- Detect the containment tree of foreground objects using holes of each frame object.
- Replace every blob with one foreground pixel.
- Replace every frame with a rectangular border padded two cells around its direct children.
- Place siblings by their original row/column overlap and order, with one empty cell between canonical objects.

Key insight: The exact wobbly geometry is irrelevant; only enclosure plus relative sibling positions define the output.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="b0039139" lines="29-44">
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

<reference_puzzle index="14" source="eval" puzzle_id="4a21e3da" lines="8-20">
<description>
Cut a sprite along border-control rays and slide pieces to grid corners.

Input structure:
- bg, fg (main object, ~centered), ctrl (1-2 pixels on grid edges).

Rule:
- Draw rays inward from each ctrl pixel along its row/col, painting only bg cells up to the farthest fg pixel (intermittent).
- Rays partition the grid; fg on the cut line stays in place.
- Cut pieces in regions touching a control edge slide to that corner; others vanish.

Key insight: Only regions sharing a border edge with a control pixel keep their piece; the ray skips fg cells.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="d35bdbdc" lines="120-134">
<description>
Transfer center colors between bordered sprites matched by color, keeping only path-connected ones.

Input structure:
- bg, grey (5) path network, and several small "bordered sprites" each having a border color and a single center pixel of a different color.
- Grey path forms a one-stroke trail under 8-connectivity; its start/end cells touch exactly two destination sprites via ports.

Rule:
- Destination sprites are determined by the grey trail's two endpoints.
- For each destination, find the source sprite whose border color equals the destination's center color.
- Replace the destination's center pixel with the source's center color.
- Delete all non-destination sprites; grey paths and destinations remain.

Key insight: Center color of each destination acts as a lookup key into the border colors of other sprites, enabling indirect color transfer.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="d8e07eb2" lines="29-47">
<description>
Look up query glyphs in a reference table and mark whether they form a line.

Input structure:
- Teal background with two full-width pink separator rows.
- Above the first separator is a query row of 5x5 glyph cells.
- Between separators is a reference table of the same 5x5 cells; a glyph is
  identified by both its color and its 3x3 shape.

Rule:
- Green-fill the background inside every table cell whose glyph exactly
  appears in the query row.
- If the matched table cells form one complete row or one complete column,
  also green-fill the query band and the bottom verdict band.
- Otherwise leave the query band unchanged and fill the bottom verdict band red.

Key insight: The bottom color is not a count; it is the yes/no answer to
whether the queried glyphs align as a full row or column in the table.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="8e5c0c38" lines="9-21">
<description>
Clean each color layer by enforcing vertical mirror symmetry.

Input structure:
- bg = background. Several non-bg colors, each forming a nearly-symmetric pattern about some vertical axis.
- Some pixels in each color are "noise" that breaks the symmetry.

Rule:
- For each non-bg color, find the vertical axis that best explains its pixel layout.
- Remove any pixel that lacks a same-color mirror counterpart across that axis.

Key insight: Each color has its own independent symmetry axis; the task is per-color denoising, not whole-grid symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="142ca369" lines="6-19">
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

<reference_puzzle index="19" source="eval" puzzle_id="abc82100" lines="70-86">
<description>
Use local source-target-pattern examples to replace matching source pixels.

Input structure:
- Black background with isolated colored pixels and teal (8) pattern components.
- A rule example is a 4-neighbor colored pair source->target whose continued
  direction reaches an anchor touching a teal pattern component.

Rule:
- For each rule example, copy that teal component onto every other pixel of
  source.color, recoloring the copied component to target.color.
- Translation maps the rule anchor onto the source pixel being replaced.
- Remove teal components and all pixels used by rule examples.

Key insight: The two adjacent colored pixels define the color substitution and
the anchor point; the nearby teal component is the stencil that gets recolored.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="dfadab01" lines="79-93">
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

