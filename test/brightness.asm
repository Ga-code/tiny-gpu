MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx             #R0 = i
CONST R1, k                        # set R1 to k
LDR R2, R0                         # load the pixel value for the thread
CONST R3, #255                   # R3 = 255
SUB R4, R3, R2                     # R4 = 255 - P
CMP R4, R1                         # compares if 255 - P is greater than k
BRzp #10                           # if it is, skips to 11
CONST R2, #255                   # sets P to 255
CONST R1, #0                       # sets k to 0 
ADD R5, R2, R1                     # R5 = new pixel value
STR R0, R5                         # overwrites pixel value with new value
RET



