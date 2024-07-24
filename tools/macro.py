import imagej
import os

# initialize ImageJ2 with Fiji plugins
ij = imagej.init('sc.fiji:fiji')
print(f"ImageJ2 version: {ij.getVersion()}")

script = """
macro "Check Slice Intensity and Process" {
    // Step 1: Check Slice Intensity and delete low-intensity slices
    setBatchMode(false);
    n = nSlices; 
    if (n > 1) {  // Only execute this part if the image is a stack
        j = 1;
        while (j <= n) {
            setSlice(j);
            run("Measure");
            mean = getResult("Mean", nResults-1);
            if (mean < 100) {
                run("Delete Slice");
                n--;
            } else {
                j++; 
            }
        }
    }

    // Step 2: Normalization and Subtraction of Average Projection
    nor = getTitle();
    setOption("ScaleConversions", true);
    run("16-bit");
    newmin = 0;
    newmean = 4000;
    for (i = 1; i <= nSlices; i++) {
        setSlice(i);
        getStatistics(area, mean, min, max, std, histogram);
        fact = newmean / mean;  
        run("Multiply...", "value=" + fact + " slice");
    }
    a = getTitle(); 
    run("Z Project...", "projection=[Average Intensity]");
    b = getTitle();
    imageCalculator("Subtract create 32-bit stack", a, b);
    or = getTitle();
    setOption("ScaleConversions", true);
    run("16-bit");
}
"""

image = ij.io().open('test_image.tif')
args = {"image":image}
