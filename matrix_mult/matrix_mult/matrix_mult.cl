__kernel void matrix_mult(__global float * A, __global float * B, __global float * C, int N, int M)
{
   int row = get_global_id(0);
   int col = get_global_id(1);

   if (row >= N || col >= N)
      return;

   int HM = (M - 1) / 2;
   float sum = 0;

   int A_row = 0;
   int A_col = 0;
   float A_cur = 0.0;
   
   for (int k = -HM; k <= HM; ++k) {
	   for (int l = -HM; l <= HM; ++l) {
		   A_row = row + k;
		   A_col = col + l;
		   A_cur = 0.0;
		   if (!(A_row < 0 || A_row >= N || A_col < 0 || A_col >= N)) {
			   A_cur = A[A_row * N + A_col];
		   }
		   sum += A_cur * B[(k + HM)*M + (l + HM)];
	   }
   }

   C[row * N + col] = sum;
}
