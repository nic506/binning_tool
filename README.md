Script run order: (1) Merging -> (2) Edge detection -> (3) Binning<br>

Merging script overview:  
  - Dilates input ROIs to decrease gaps between boundaries  
  - Generates a binary PNG mask, merging nearby ROIs  
  - Erosion of mask to counteract dilation, restoring approx original size  
  - Converts PNG mask to single ROI
  
Edge Detection script overview:  
  - Assumes that ROIs are rectangle-like polygons (4 rough vertices) and that binning is desired along longest edge and its opposite edge  
  - For each ROI anchor point:  
      Measures angles at multiple distances along polygon perimeter;  
      Scores how often angles fall within "corner-like" range (e.g. 30–150°);  
      Computes angle consistency (standard deviation);  
      Clusters nearby anchor points, keeping the best-scoring most consistent one;  
  - Selects the top 4 vertex candidates based on score and consistency  
  - Extracts the longest edge between candidate vertices and the edge opposite to it  
### AUTOMATED EDGE DETECTION (above) FAILS FOR EXTREME CASES, SO CURRENTLY DEVELOPING A BETTER USER INTERFACE TO MANUALLY SELECT VERTICES ###  
  
Binning script overview:  
  - Vertical binning: Intersects tangents from one edge to the other, spaced at a defined width in μm  
  - Vertical + Horizontal binning: Divides each vertical bin horizontally into an equal number of segments  
  - Horizontal binning: Combines vertical bins within the same horizontal band  
  - Outputs a .zip file containing all bin ROIs  
