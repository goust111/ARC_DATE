你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="898e7135" lines="7-20">
<description>
Inflate a frame's hole-layout and fill each hole with its matching piece.

Input structure:
- bg = most common color. Largest component is a monochrome frame enclosing bg holes.
- Outside pieces composed of k×k blocks; k = side of smallest solid square piece.

Rule:
- Extract connected hole regions inside the frame as slots.
- Match each piece to a slot by comparing block-level shapes under rotation/flip.
- Output the layout scaled by k, with pieces placed in slots; frame color as background.

Key insight: Pieces are k×k-block-scaled versions of holes; downsampling reveals shape matches under all 8 orientations.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="f931b4a8" lines="10-27">
<description>
Count the two upper quadrant control regions to size an output made from the lower quadrant pattern.

Input structure:
- The grid is an even square split into four equal quadrants.
- Black is the background.
- The upper-left and upper-right quadrants are control regions.
- The lower-left quadrant is a color source tile; the lower-right quadrant is either a stencil tile or empty.

Rule:
- Output height is the number of non-background pixels in the upper-left quadrant.
- Output width is the number of non-background pixels in the upper-right quadrant.
- If the lower-right stencil has foreground pixels, tile it over the output; each stencil background pixel is filled from the lower-left quadrant cell at the corresponding coarse tile position.
- If the lower-right stencil is empty, tile the lower-left quadrant directly over the output.

Key insight:
- The top quadrants specify only dimensions, while the bottom quadrants specify a two-level repeated pattern: a fine stencil and coarse fill colors.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="4c416de3" lines="99-118">
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

<reference_puzzle index="4" source="eval" puzzle_id="6e4f6532" lines="26-41">
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

<reference_puzzle index="5" source="eval" puzzle_id="abc82100" lines="70-86">
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

<reference_puzzle index="6" source="eval" puzzle_id="b0039139" lines="29-44">
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

<reference_puzzle index="7" source="eval" puzzle_id="80a900e0" lines="69-82">
<description>
Extend diagonal rays outward from outer-border segments adjacent to diamond shapes on a checkerboard.

Input structure:
- Checkerboard bg (two most common colors).
- Hollow diamond outlines (8-pixel) in a non-bg color (inner border).
- Short diagonal line segments in another non-bg color adjacent to each diamond edge (outer border).

Rule:
- For each outer-border segment, determine the outward direction (perpendicular to the segment, away from the diamond).
- From each endpoint of the segment, cast a diagonal ray outward, painting over bg pixels with the segment's color until hitting a non-bg pixel or the grid edge.

Key insight: Rays emit only from the two endpoints of each outer-border segment, not every pixel, and travel diagonally away from the diamond.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="c7f57c3e" lines="22-35">
<description>
Swap every sprite to the opposite prototype while preserving scale and position.

Input structure:
- bg = most common color. Two distinct small sprite patterns (prototypes) appear at various scales.
- Each sprite instance is a scaled copy of one of the two prototypes, placed on the grid.

Rule:
- Identify the two prototype patterns from the smallest-area sprites.
- For each sprite, determine which prototype it matches and its integer scale factor.
- Replace it with the other prototype at the same scale, aligned by an internal anchor region (color-1 pixels).

Key insight: Every sprite toggles to the alternate prototype; the anchor sub-region inside each sprite pins the alignment so swapped sprites land in the correct position regardless of scale.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="db0c5428" lines="32-51">
<description>
Expand a 3x3 block motif into a 5x5 block motif around its missing center.

Input structure:
- Teal background surrounds one 9x9 foreground motif.
- The motif is a 3x3 array of 3x3 blocks; the middle block is empty teal.
- Colored pixels immediately around the empty middle block encode the border
  of the missing center block.

Rule:
- Keep the original 9x9 motif in place.
- Copy each of the eight surrounding 3x3 blocks to the opposite outer slot
  of the implied 5x5 block grid, without rotating it.
- Fill the center block from the eight pixels around the hole; its center is
  the motif's majority foreground color.

Key insight: The visible 9x9 motif is the middle of a larger periodic
block-level pattern, and the empty center is reconstructed from its immediate
inner boundary.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="136b0064" lines="25-39">
<description>
Draw a polyline by decoding 3x3 glyph instructions read across a divided grid.

Input structure:
- Left of a full-height vertical divider: 3x3 colored glyphs on bg.
- Right of divider: a single gray (5) start pixel.

Rule:
- Remove the divider; shift the gray start into the left canvas.
- Read glyphs in column-major order (left→right, top→bottom).
- Each glyph's shape (pixel area) encodes a segment: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
- Append segments sequentially, stepping down one cell between each.

Key insight: Glyph pixel count (area) uniquely determines both segment length and direction — the shape itself is the instruction.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="78332cb0" lines="10-23">
<description>
Reorder 5×5 blocks into a connected strip by matching edge foreground pixels.

Input structure:
- bg = background; sep = divider color (full row/column lines).
- Several 5×5 sprite blocks in a grid, separated by sep lines.

Rule:
- Two blocks "connect" when their facing edges share foreground pixels at the same offset (e.g., A's right edge ∩ B's left edge).
- Chain all blocks in the orientation (horizontal or vertical) that admits a complete sequence.
- Output is a 1×N strip of blocks in chain order, separated by sep lines.

Key insight: Edge-contact overlap determines both block ordering and output orientation.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="eee78d87" lines="40-55">
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

<reference_puzzle index="13" source="eval" puzzle_id="a395ee82" lines="9-24">
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

<reference_puzzle index="14" source="eval" puzzle_id="9385bd28" lines="71-87">
<description>
Fill rectangles between diagonal marker pairs using a color legend.

Input structure:
- bg grid with a vertical legend at the bottom-left: each row is a (marker_color, fill_color) pair.
- Scattered L-shaped corner markers of each marker color define opposite corners of axis-aligned rectangles.
- Some markers are solid 2×2 blocks instead of corner pairs.

Rule:
- Read legend bottom-to-top; for each entry find all pixels of that marker color.
- Determine which two bbox corners carry L-shaped anchors — the *other* diagonal defines the fill region.
- Fill the diamond (all shortest-path cells between those two corners) with the legend's fill color.
- Solid 2×2 marker blocks are simply recolored to the fill color.
- Later legend entries are protected from earlier fills.

Key insight: The L-shaped anchors sit on the marker diagonal; the fill goes on the opposite diagonal of the bounding box.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="28a6681f" lines="8-23">
<description>
Simulate water pixels falling under gravity into basins formed by colored obstacles.

Input structure:
- bg = 0. Water pixels = color 1 (blue).
- Other colors form static solid structures (walls, steps, containers).
- Water starts in unstable positions above or beside basins.

Rule:
- Each water pixel falls straight down through bg cells.
- When blocked below, it slides diagonally through the unique open side cell.
- When resting on a surface, it slides horizontally toward the nearest gap where it can fall again.
- Process repeats until no water pixel moves.

Key insight: Water seeks the deepest reachable basin via diagonal and horizontal sliding, not just vertical fall.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="65b59efc" lines="9-24">
<description>
Decode a stamp map: read templates, markers, and palette from a tiled grid, then render stamps.

Input structure:
- Grid divided into K tile columns (each N wide) separated by 1-col dashed separators.
- Top band (rows 0..N-1): K template masks, one per tile column.
- Middle band (rows N+1..2N): marker pixels whose color picks a template.
- Bottom band (rows 2N+2..): K palette swatches giving the output color per template.
- Separator rows at rows N and 2N+1 (dashed pattern of sep color).

Rule:
- Each marker's local position (lr, lc) within its tile column determines which N×N output block to stamp.
- The marker's color identifies which template to use; the palette recolors the template.

Key insight: Marker color encodes template index; local position within tile column encodes stamp block.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="53fb4810" lines="105-119">
<description>
Extend marker tiles outward from '+' shapes by repeating them to the grid edge.

Input structure:
- bg = most common color.
- A dominant non-bg color forms several '+'-shaped objects.
- Small colored patches (1–few cells) sit adjacent to each arm of a '+'.

Rule:
- For each marker tile touching a '+', determine which side of the '+' it sits on.
- Project the tile away from the '+' in that direction, repeating the tile pattern periodically until the grid boundary.
- '+' pixels are never overwritten.

Key insight: Each marker's attachment side determines its extension direction; the pattern repeats periodically, not as a single copy.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="5545f144" lines="191-207">
<description>
Move a triangular sprite across the grid, consuming aligned marker pixels.

Input structure:
- Frames concatenated horizontally with solid divider columns; bg is most common color.
- Each frame: a triangular arrow-shaped sprite and scattered single-pixel markers, all one color.

Rule:
- Start from the rightmost frame.
- At each step, exactly one cardinal direction from the head has reachable
  markers (on a half-line: same row or column); the sprite advances to the
  nearest one.
- On reaching a marker, consume it; sprite re-faces that direction.
- Repeat until no markers remain; output the final sprite position on a clean bg.

Key insight: The triangle's tip encodes facing direction, which sets marker-consumption priority.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="dbff022c" lines="275-288">
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

<reference_puzzle index="20" source="eval" puzzle_id="7666fa5d" lines="12-26">
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

