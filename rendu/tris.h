/* Merge sort */
void merge_sort(int tab[], int borne_inf, int borne_sup);

void merge_sort_omp(int tab[], int borne_inf, int borne_sup);

/* Quick sort */
void quick_sort(int tab[], int tab_size);

void quick_sort_omp(int tab[], int tab_size);

/* Bubble sort */
void bubble_sort(int tab[], int tab_size);

void bubble_sort_omp(int tab[], int tab_size);

/* Merge two arrays already sorted */

void merge(int sorted[], int left, int left_size, int right, int right_size);

void merge_tabs(int sorted[], int left[], int left_size, int right[], int right_size);