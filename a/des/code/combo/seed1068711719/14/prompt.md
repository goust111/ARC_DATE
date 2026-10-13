你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="16b78196" lines="6-20">
<description>
Assemble colored sprites into a central bar's notches, chaining successive pieces along fixed-width strips.

Input structure:
- bg = most common color. Largest monochrome component is a solid "bar" with boundary notches (bg holes opening to its bbox edge).
- Several smaller colored sprites scattered around the bar.

Rule:
- Each bar notch is filled by a root piece whose shape exactly covers the hole.
- The root defines a strip (column-span or row-span) for its chain.
- Successive pieces fill the bg cells inside the previous piece's bbox, locked to the same strip width.
- Multiple chains grow independently from different bar notches.

Key insight: The root piece's extent perpendicular to the chain direction locks the strip width for all subsequent pieces in that chain.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="45a5af55" lines="5-18">
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

<reference_puzzle index="3" source="eval" puzzle_id="21897d95" lines="283-301">
<description>
Repaint regions from T-shaped control markers, erase the markers, and optionally rotate based on a color-9 L-marker.

Input structure:
- The grid is a mosaic of monochrome regions.
- Color 1 forms T-shaped control markers on region boundaries.
- Some controls use three singleton color-1 pixels around a colored center; that center is a literal replacement color.
- An optional 5-cell L-shaped color-9 marker triggers 90-degree rotation of the output.

Rule:
- For a color-1 T marker, the single stem points at the destination region to repaint.
- The two diagonal cells behind the stem identify the source region; the destination gets that source region's original color.
- For a colored-center T marker, the same stem identifies the destination, but the center color is the replacement.
- All marker cells are filled with the final color of the source-side region.
- A 5-cell L-shaped color-9 marker controls rotation: ┐ (corner top-right) → CCW 90°, ┌ (corner top-left) → CW 90°.

Key insight:
- The stem chooses what changes, while the two diagonals behind the stem choose where the marker itself belongs.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="963c33f8" lines="86-108">
<description>
Drop the columns of the top marker downward, stopping blue-ended columns on grey obstacles and letting maroon-ended columns pass to the bottom.

Input structure
- The background is the dominant color.
- The most numerous non-background color forms stationary grey obstacle pixels.
- The other two colors form a single 3x3 top-edge marker whose top two rows
  are maroon and whose bottom row contains maroon or blue control pixels.

Rule
- Erase the top marker.
- Treat each marker column as a vertical drop retaining its three colors.
- A column whose bottom control pixel is blue stops immediately above the
  first grey pixel below that column, or bottom-aligns if none exists.
- Contiguous maroon-ended columns pass through obstacles and bottom-align;
  if such a run has a clear down-right slide from a grey ledge, shift the
  run one column right before bottom-aligning.
- When a maroon-ended drop lands through a grey bar on the bottom edge, clear
  the rest of the marker-width gap in that bottom bar.

Key insight
- The bottom row of the top marker is a control strip: blue cells mean
  "stick to the first obstacle," while maroon cells mean "fall through."
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="833966f4" lines="8-22">
<description>
Swap the upper two-pixel segment and the lower two-pixel segment around the fixed center pixel.

Input structure:
- The grid is a 5x1 vertical column of individual colored pixels.
- The middle pixel is a separator between two same-size two-pixel segments.

Rule:
- Keep the middle pixel in place.
- Reverse the order of the two pixels above the middle.
- Reverse the order of the two pixels below the middle.

Key insight:
- The transformation is local vertical order reversal on each side of the center, not a color-specific rule.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="95755ff2" lines="45-60">
<description>
Paint straight rays from colored side markers into the hollow diamond frame until an interior foreground pixel blocks each ray.

Input structure
- The background contains one complete one-pixel-thick diamond frame.
- Colored marker pixels outside the diamond sit on the top, bottom, left, or right side of the frame's bounding box.
- Any non-background, non-frame pixels inside the diamond are blockers that remain unchanged.

Rule
- Each outside marker emits inward perpendicular to its side of the diamond bounding box.
- Along that row or column, paint only background pixels strictly inside the diamond with the marker color.
- Stop a ray when it reaches an existing interior foreground pixel; keep the frame and all original colored pixels unchanged.

Key insight
- The diamond frame defines a strict interior, while the side markers are column or row emitters whose color fills the visible interior interval until blocked.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="89565ca0" lines="120-136">
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

<reference_puzzle index="8" source="train" puzzle_id="7acdf6d3" lines="25-42">
<description>
Move the source-color pixels into the unique outline interior whose size matches the source count.

Input structure
- The grid has a dominant background and exactly two foreground colors.
- One foreground color forms 8-connected outline objects with row-wise interior cells.
- The other foreground color supplies a set of source pixels away from the chosen outline.
Rule
- For each possible outline color, compute each 8-connected outline object's
  row-span interior: background pixels strictly between the leftmost and
  rightmost outline pixels on each occupied row.
- Select the unique outline interior whose cell count equals the total
  number of pixels of the other foreground color.
- Erase the source pixels and paint that matched interior with the source color.
Key insight
- The colors are roles, not fixed IDs: the source color is identified by
  counting, and the destination is the outline with a matching interior area.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="e734a0e8" lines="44-59">
<description>
Copy the unique colored sprite tile into every tiled cell marked by an internal separator-colored pixel.

Input structure
- Full same-color separator rows and columns partition the grid into equal rectangular tiles.
- One tile contains the only non-background, non-separator colored sprite.
- Target tiles are otherwise background-colored but contain at least one internal pixel of the separator color.

Rule
- Keep the separator lines and the source sprite tile unchanged.
- Replace each marked target tile with an exact copy of the source tile.
- Unmarked background-only tiles remain unchanged.

Key insight
- The internal separator-colored pixels are placement markers; the single colored tile is the template to stamp into those marked cells.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="aaecdb9a" lines="10-26">
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

<reference_puzzle index="11" source="train" puzzle_id="57edb29d" lines="88-103">
<description>
Extract the single plain rectangle and redraw it with the aligned rectangles' border-relative marker pattern.

Input structure:
- The boundary color is the background/separator.
- Foreground objects are dense rectangular regions; exactly one rectangle has only its base color.
- Other rectangles contain their base color plus marker pixels.

Rule:
- Crop the only plain rectangle; this crop is the output canvas.
- Use marked rectangles whose row span overlaps the plain rectangle's row span.
- Transfer their marker pixels by preserving each marker run's distance from the nearest rectangle edges.

Key insight:
- The marked rectangles are examples of an edge-relative pattern, so the missing plain rectangle receives the same pattern resized to its own dimensions.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="bc93ec48" lines="48-63">
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

<reference_puzzle index="13" source="train" puzzle_id="bae5c565" lines="10-28">
<description>
Erase the top palette row and redraw it below as a widening wedge centered on its missing cell.

Input structure
- The top row is a palette row with exactly one background gap.
- Below it, the only foreground is one vertical marker bar of a single color.
- The marker bar is either in the gap column or immediately beside it.

Rule
- Clear the grid to background.
- Starting at the top of the marker bar, each next row copies one more
  column of the top-row palette on both sides of the gap.
- Fill the gap column with the marker color, and keep the vertical marker bar.

Key insight
- The lone background cell in the top row is the center of the expanding
  copy; the lower vertical bar provides the center color and the row where
  the expansion begins.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="7e4d4f7c" lines="9-24">
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

<reference_puzzle index="15" source="train" puzzle_id="d968ffd4" lines="15-28">
<description>
Extend each of two aligned rectangular marker objects toward the other, filling half of the gap across the full perpendicular span.

Input structure
- A dominant background contains exactly two solid, same-sized monochromatic rectangles.
- The rectangles share either the same row span or the same column span, with a background gap between their facing edges.

Rule
- If the rectangles are side-by-side, color the left half of the inter-object gap across every row with the left object's color, and the right half with the right object's color.
- If one rectangle is above the other, color the upper half of the gap across every column with the upper object's color, and the lower half with the lower object's color.
- When the gap length is odd, leave the center row/column as background.

Key insight
- The marker objects are endpoints of an axial corridor; their own size identifies the facing edge, while the filled corridor spans the whole grid perpendicular to that axis.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="2e65ae53" lines="88-103">
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

<reference_puzzle index="17" source="train" puzzle_id="a834deea" lines="20-39">
<description>
Fill each black 5x5 region's missing interior positions with a fixed 3x3 positional color legend.

Input structure
- The grid uses teal as the dominant background and black for one or more
  5x5 regions.
- Each black region has a black one-pixel border and a 3x3 interior that
  may already contain teal pixels.

Rule
- Overlay the legend 1 7 6 / 4 8 5 / 2 9 3 on the 3x3 interior of every
  5x5 black region.
- Wherever an interior non-center legend position is still black, recolor it
  to the corresponding legend color; leave existing teal pixels, the center,
  and the black border unchanged.

Key insight
- The colored output pixels are not derived from the teal motif shape; they
  are the fixed position labels for the missing cells of each 3x3 interior.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="522fdd07" lines="8-23">
<description>
Cycle each solid odd-sided square to the next smaller centered square, wrapping a single pixel to a 9x9 square.

Input structure:
- The grid has one dominant background color.
- Every foreground object is a separated 4-connected monochrome solid square.
- Square side lengths are odd and belong to the cycle 1, 3, 5, 7, 9.

Rule:
- Clear the canvas to the background.
- Keep each square's color and center.
- Replace side length n by n - 2 when n > 1; replace side length 1 by side length 9.

Key insight:
- The visible operation is on square size, not color: side lengths cycle 1 -> 9 -> 7 -> 5 -> 3 -> 1.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="13e47133" lines="56-69">
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

<reference_puzzle index="20" source="eval" puzzle_id="78332cb0" lines="10-23">
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
11. 可以少出现以下三种类型的创新：查表填充、计数→索引、箭头→方向+距离；多些其他创新组合。

生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。
