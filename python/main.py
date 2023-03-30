import easyocr
import numpy
import math

###########################################################################################################################
# Functions
def drawHistogram(mat):
    #i1 is the intensity between 0 and 63, i2 is between 64 and 123, i3 is between 124 and 193, i4 is between 194 and 255 
    i1 = 0
    i2 = 0
    i3 = 0 
    i4 = 0

    for row in mat:
        for num in row:
            if num >= 0 and num <= 63: i1 += 1
            elif num >= 64 and num <= 123: i2 += 1
            elif num >= 124 and num <= 193: i3 += 1
            elif num >= 194: i4 += 1

    print("Histogram")
    print("I1 {}, I2 {}, I3 {}, I4 {}".format(i1, i2, i3, i4))

def gammaTransformation(a, y, mat):
    for i in range (4):
        for j in range (4):
            num = a * math.pow(mat[i, j], y)
            mat[i, j] = num if num < 256 else 255
            

    print("after gamma transformation with y {} and a {}".format(y, a))
    print(mat)
    print("Lmax {}, Lmin {}, Lmean {}".format(mat.max(), mat.min(), int(mat.mean())))
    drawHistogram(mat)
    print("--------------------------------------------------------------------------", "\n")   

def logTransformation(mat):
    for i in range (4):
        for j in range (4):
            #print(type(matrix[i, j]))
            mat[i, j] = math.log10(mat[i, j] + 1)

    print("after log transformation")
    print(mat)
    print("Lmax {}, Lmin {}, Lmean {}".format(mat.max(), mat.min(), int(mat.mean())))
    drawHistogram(mat)
    print("--------------------------------------------------------------------------", "\n")   
#--------------------------------------------------------------------------------------------------------------------------

###########################################################################################################################
# Main
def main():
    # Use easyocr to read from image 
    reader = easyocr.Reader(['en'], gpu=False) # this needs to run only once to load the model into memory
    results = reader.readtext("../images/matrixI1.jpg", detail = 0)

    # Sometimes easyocr puts a space between digits so we replace it with ""
    # Convert str to int
    for i in range(16):
        results[i] = int(results[i].replace(" ", "")) 

    # Convert to numpy array
    results = numpy.array(results)

    # Reshape into matrix
    matrix = results.reshape(4, 4)
    print("original matrix")
    print(matrix, "\n")
    print("--------------------------------------------------------------------------", "\n")   

    # Why matrix is passed like a copy -> https://stackoverflow.com/questions/47893884/numpy-array-pass-by-value
    # Logarithmic transformation
    logTransformation(matrix.copy())

    # Gamma transformation
    gammaTransformation(1, 0.7, matrix.copy())
    gammaTransformation(1, 1.8, matrix.copy())
    gammaTransformation(5, 0.7, matrix.copy())
#--------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
















