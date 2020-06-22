#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>


static PyObject* step(PyObject *self, PyObject *args);

static PyMethodDef methods[] = {
    {"step", step, METH_VARARGS, "Perform a single iteration of ViBe algorithm."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vibe_step = {
    PyModuleDef_HEAD_INIT,
    "vibe_step",
    "C implementation of a single iteration of ViBe algorithm.",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_vibe_step(void) {
    return PyModule_Create(&vibe_step);
}

struct location {
    int i;
    int j;
};

inline struct location get_neib(int i,int j,int H,int W) {
    int n_i, n_j;
    int t = rand() % 8;
    int seq[] = {0,1,1,1,0,-1,-1,-1};

    n_i = i + seq[t];
    n_j = j + seq[(t + 6) % 8]; 
    
    if (n_i < 0) {n_i = 1;}    
    if (n_j < 0) {n_j = 1;}
    if (n_i >= H) {n_i = H - 2;}
    if (n_j >= W) {n_j = W - 2;}

    struct location out;
    out.i = n_i;
    out.j = n_j;
    return out;
}

typedef unsigned char byte;

static PyObject* step(PyObject *self, PyObject *args) {
    PyArrayObject *image, *model, *seg_map;
    int R, n_min, subsamp;

    if (!PyArg_ParseTuple(args, "OOOiii", &image, &model, &seg_map, &R, &n_min, &subsamp)) {return NULL;}
    
    int R2 = R*R;

    int H, W, N;
    H = image->dimensions[0];
    W = image->dimensions[1];
    N = model->dimensions[2];

    srand(time(NULL));

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            byte r = *(byte*)(image->data + i*image->strides[0] + j*image->strides[1] + 0*image->strides[2]);
            byte g = *(byte*)(image->data + i*image->strides[0] + j*image->strides[1] + 1*image->strides[2]);
            byte b = *(byte*)(image->data + i*image->strides[0] + j*image->strides[1] + 2*image->strides[2]);

            byte *model_ptr = (byte*)(model->data + i*model->strides[0] + j*model->strides[1]);

            int count = 0, index = 0;
            while ((index < N)&&(count < n_min)) {
                int dr = (int)r - (int)*(model_ptr + index*model->strides[2] + 0*model->strides[3]);
                int dg = (int)g - (int)*(model_ptr + index*model->strides[2] + 1*model->strides[3]);
                int db = (int)b - (int)*(model_ptr + index*model->strides[2] + 2*model->strides[3]);

                if (dr*dr + dg*dg + db*db < R2) {count++;}

                index++; 
            }

            if (count >= n_min) {
                int rnd;

                rnd = rand() % subsamp;
                if (rnd == 0) {
                    rnd = rand() % N;

                    *(model_ptr + rnd*model->strides[2] + 0*model->strides[3]) = r;
                    *(model_ptr + rnd*model->strides[2] + 1*model->strides[3]) = g;
                    *(model_ptr + rnd*model->strides[2] + 2*model->strides[3]) = b; 
                }

                rnd = rand() % subsamp;
                if (rnd == 0) {
                    rnd = rand() % N;

                    struct location neib = get_neib(i,j,H,W);
                    byte *neib_ptr = (byte*)(model->data + neib.i*model->strides[0] + neib.j*model->strides[1]);

                    *(neib_ptr + rnd*model->strides[2] + 0*model->strides[3]) = r;
                    *(neib_ptr + rnd*model->strides[2] + 1*model->strides[3]) = b;
                    *(neib_ptr + rnd*model->strides[2] + 2*model->strides[3]) = g;
                }

                *(byte*)(seg_map->data + i*seg_map->strides[0] + j*seg_map->strides[1]) = 0;
            } else {
                *(byte*)(seg_map->data + i*seg_map->strides[0] + j*seg_map->strides[1]) = 255;
            }
        }
    }

    return Py_None;
}
