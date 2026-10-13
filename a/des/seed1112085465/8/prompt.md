你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="edcc2ff0" lines="47-63">
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

<reference_puzzle index="2" source="train" puzzle_id="9968a131" lines="10-27">
<description>
Shift alternating two-color row bars so each row aligns to the topmost bar's color columns.

Input structure:
- The dominant background surrounds horizontal two-pixel bars.
- Every foreground row has exactly one adjacent bar using the same two colors.
- The topmost bar gives the reference left/right column for each color.

Rule:
- Keep bars whose left color is already in its reference column.
- For a reversed bar, shift the whole two-pixel bar horizontally until its
  left color lands in that color's reference column.
- Background cells vacated by shifted bars become background.

Key insight:
- The rows are not recolored; each two-color bar is translated so equal
  colors line up with the topmost reference bar.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="fd02da9e" lines="26-41">
<description>
Replace each colored corner marker with its corner-specific inward stencil in the same color.

Input structure
- The grid is 8x8 with a single dominant background color.
- Foreground pixels are marker pixels, each located at one of the four corners.
- A marker's color is carried to the stencil selected by its corner position.

Rule
- Erase each corner marker back to the background.
- Top corner markers become an inward 2x2 block one cell away from the top and side edges.
- Bottom corner markers become the mirrored three-pixel inward bent stencil shown by the bottom-corner examples.

Key insight
- The marker position selects the fixed stencil; the marker color only supplies the paint color, so multiple corners apply independently.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="16b78196" lines="6-20">
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

<reference_puzzle index="5" source="train" puzzle_id="aa62e3f4" lines="18-35">
<description>
Replace the outlined object with the outside contour of that outline in the marker color.

Input structure:
- A dominant background surrounds one 8-connected foreground outline.
- The outline is the unique largest foreground color.
- One or more smaller interior colors are inside the outline; the marker color
  is the unique smallest non-outline color.

Rule:
- Erase all original foreground pixels to background.
- Paint every background pixel that is 4-adjacent to the outline with the
  marker color.
- Keep the grid shape unchanged.

Key insight: The colored interior marker supplies the paint color, while the
largest outline supplies only the contour whose outside edge is traced.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="5b37cb25" lines="59-73">
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

<reference_puzzle index="7" source="train" puzzle_id="dc46ea44" lines="11-25">
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

<reference_puzzle index="8" source="train" puzzle_id="9473c6fb" lines="10-25">
<description>
Recolor the sparse foreground pixels in scan order along their longer spread, cycling red, teal, and grey.

Input structure:
- The dominant color is the background.
- Every foreground object is a single isolated 4-connected pixel.
- The foreground pixel set has a rectangular bbox whose row or column extent defines the scan direction.

Rule:
- Keep the background unchanged.
- If the foreground bbox is at least as tall as it is wide, order foreground pixels row-major; otherwise order them column-major.
- Recolor the ordered pixels by repeating red, teal, grey from the first pixel onward.

Key insight:
- The original foreground colors are only markers; the output color depends on each marker's ordinal position in the dominant scan direction.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="d753a70b" lines="81-93">
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

<reference_puzzle index="10" source="eval" puzzle_id="5dbc8537" lines="10-23">
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

<reference_puzzle index="11" source="train" puzzle_id="f341894c" lines="63-78">
<description>
Reorient each two-pixel blue/pink segment so it reads pink then blue when viewed from its border marker.

Input structure
- The background is dominant, with one border-only marker color and blue/pink foreground segments.
- Every marker lies on the outer border and points inward along its row or column.
- Along exactly one inward ray from each marker is a single adjacent two-pixel blue/pink segment.

Rule
- For each marker, look inward perpendicular to the border side containing the marker.
- Rewrite the two blue/pink pixels on that ray so the nearer one is pink and the farther one is blue.
- Keep marker pixels, background pixels, and all other pixels unchanged.

Key insight
- The marker gives a local viewing direction; the segment's required order is relative to that marker, not to global row or column order.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="b7955b3c" lines="386-403">
<description>
Replace gray plus-shaped damage masks with the hidden colors of the rectilinear mosaic.

Input structure:
- Color 8 appears only as isolated 3x3 plus masks.
- The non-mask pixels are visible parts of a clean grid made from horizontal
  and vertical color regions.

Rule:
- Treat every color-8 pixel as unknown damage and leave all other pixels fixed.
- First copy unknown pixels from exact matching parallel rows or columns.
- Then fill any remaining mask cells from opposite-side continuations and the
  better-supported local straight edge.

Key insight:
- The gray pluses are occluders, not objects; the underlying colors are
  determined by repeated line templates and nearby axis-aligned boundaries.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="5623160b" lines="44-59">
<description>
Push every colored object touching maroon markers straight outward until it reaches the grid edge.

Input structure:
- The background is the dominant color.
- Maroon pixels are fixed markers.
- Every other foreground 4-connected object touches the marker pixels on exactly one cardinal side.

Rule:
- Clear each non-background, non-maroon object from its original position.
- Translate it in the direction away from its maroon contact until the object reaches the grid boundary.
- Keep the maroon marker pixels unchanged.

Key insight:
- The maroon contact side is a launcher: it determines a unique outward ray for the whole object, and the edge of the grid determines the translation distance.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="af726779" lines="17-35">
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

<reference_puzzle index="15" source="train" puzzle_id="a6953f00" lines="45-63">
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

<reference_puzzle index="16" source="eval" puzzle_id="6e453dd6" lines="39-52">
<description>
Slide black shapes rightward to a grey separator, then shoot red rays from enclosed holes.

Input structure:
- A vertical grey (5) separator column divides the grid into left and right halves.
- Black shapes sit on the left side against a background color (bg).

Rule:
- Translate each black connected component rightward until its right edge is flush with the separator.
- After sliding, find enclosed bg regions (holes) within the black shapes that touch the column just left of the separator.
- For each row spanned by such a hole, emit a red ray across the entire right half of the grid.

Key insight: Only holes adjacent to the separator trigger rays — the sliding creates enclosures that act as "nozzles" projecting rightward.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="9841fdad" lines="15-39">
<description>
Expand the colored runs from the left reference panel into the blank right panel.

Input structure:
- A uniform rectangular frame surrounds the grid.
- One full-height divider of the frame color splits a left reference panel
  from a blank right panel.
- Both panels have one background margin column on each side of their active
  columns; the left active columns contain two-column endpoint slots, possibly
  connected by same-color horizontal runs.

Rule:
- Keep the original grid and paint only in the right panel's active columns.
- A source run exactly matching the left endpoint slot is copied to the left
  two columns of the right active span.
- A source run exactly matching the right endpoint slot is copied to the right
  two columns of the right active span.
- A source run connecting the endpoint slots is stretched across the entire
  right active span.

Key insight:
The left panel is a compressed legend for the right panel: endpoint markers
stay two pixels wide, while connectors between endpoints expand to fill the
distance between them.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="83eb0a57" lines="72-86">
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

<reference_puzzle index="19" source="train" puzzle_id="bd283c4a" lines="7-23">
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

<reference_puzzle index="20" source="train" puzzle_id="8886d717" lines="43-63">
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
