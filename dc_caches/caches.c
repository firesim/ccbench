////////////////////////////////////////////////////////////////////////////////
// Caches (time a pointer chase to get cache capacity, cache acces latency) 
// Author: Christopher Celio
// Date  : 2010 Nov 15
//
// Designed to assist in emperically discovering the cache size of a processor
// by repeatedly running this kernel with varying input sizes. 
//
// Create an array of size <input>, where each array element is a pointer to
// another element in the array (circular linked list). Arrange pointers
// randomly to minimize cache-prefetching effects, and walk the array for set
// period of elements to normalize for work/array-size. Thus, any increase in
// run-time should be localized to caching effects.
//
// The input is (at the moment) in "number of elements in the array", 
// which will be rounded to the nearest cache-line amount. This means that
// the graphing code should pay attention to the report file, and not the 
// input file when looking at where to plot a data point. 

//#define DEBUG 
//#define PRINT_ARRAY
#define PRINT_SCRIPT_FRIENDLY
 
//force benchmark to run for some minimum wall-clock time
// advantages: hopefully smoothes out noise of tests that run too quickly
// disadvantages: adds additional code to critical loop (empirically unnoticable)
#define USE_MIN_TIME
//min time is in (seconds) 
#ifndef __riscv
#define MIN_TIME (1.0)
#else
#define MIN_TIME (0.0025)
#endif
                      

// stride in # of elements, most processors have a 64byte cache line, so 16 elements
// would stride across cache lines.
#define CACHELINE_IN_BYTES_SZ (256)
//#define CACHELINE_IN_BYTES_SZ (64)
#define CACHELINE_SZ (CACHELINE_IN_BYTES_SZ/sizeof(uint64_t))

// in # bytes
#define PAGE_SZ (4096) 

#define EXTENT_SZ (1L << 30)

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <cctimer.h>
#include <cclfsr.h>
#include <math.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

// Global Variables
uint64_t  g_num_cores;
uint64_t  g_num_elements;  
uint64_t  g_num_iterations;
uint64_t  g_performed_iterations;
uint64_t  g_macaddr_suffix;
uint64_t  g_start_extent;
uint64_t  g_num_servers;

int g_stride;
int g_run_type;  // choose between stride size, or random stride 

double volatile run_time_ns;
double volatile run_time_us;
double volatile run_time_ms;
double volatile run_time_s;


// Function Declarations
int initializeGlobalArrays(uint64_t* arr_n_ptr, uint64_t num_elements, uint64_t stride);
int threadMain();

int printArray(uint64_t *arr_ptr, uint64_t num_elements, uint64_t stride);
int verifyArray(uint64_t *arr_ptr, uint64_t num_elements, uint64_t stride);

int main(int argc, char* argv[])
{
#ifdef DEBUG  
   printf("\nBegin Test\n");
#endif

   if (argc != 7)
   {
      fprintf(stderr, "argc=%d\n", argc);
      fprintf(stderr, "\n[Usage]: cache_size <Number of Array Elements (Total)> <Number of Iterations> <RunType> <MAC Addr Suffix>\n\n");
      fprintf(stderr, "   RunType encodes the stride (in 32b words) between each memory access. If RunType==0,\
                \n   then a random memory access (on cacheline boundaries) is generated\n\n");
      return 1;
   }

   g_num_elements   = atol(argv[1]);
   g_num_iterations = atol(argv[2]);
   g_run_type       = atoi(argv[3]);
   g_macaddr_suffix = atol(argv[4]);
   g_start_extent   = atol(argv[5]);
   g_num_servers    = atol(argv[6]);


#ifdef DEBUG
   fprintf(stderr, "Size of the array     = %ld\n",   g_num_elements);
   fprintf(stderr, "Number of Iterations  = %ld\n",   g_num_iterations);
   fprintf(stderr, "Stride size (run type)= %ld\n",   g_run_type);
   fprintf(stderr, "MAC address suffix    = %lx\n",   g_macaddr_suffix);
   fprintf(stderr, "Starting extent       = %ld\n",   g_start_extent);
   fprintf(stderr, "Num servers           = %ld\n\n", g_num_servers);
#endif

   if(g_run_type > 0)
      g_stride = g_run_type;

   // random access should hit cache-line addresses, to eliminate
   // spatial locality and force more misses.
   if(g_run_type == 0)
   {
      g_stride = CACHELINE_SZ;
   }

   // make the array size a multiple of the stride; round up
   // actually stride*2 so we can insure that no consecutive accesses are adjacent, and this makes implementing this easier
   if (g_num_elements % (2*g_stride) != 0)
      g_num_elements = g_num_elements + ((2*g_stride) - (g_num_elements % (2*g_stride)));

#ifdef DEBUG
   printf(" Adjusted Num of Elements: %d\n\n", g_num_elements);
#endif

   // this volatile ret_val is crucial, otherwise the entire run-loop 
   // gets optimized away! :(
   uint64_t volatile ret_val = threadMain();  

#ifdef PRINT_SCRIPT_FRIENDLY
   fprintf(stdout, "App:[caches],NumThreads:[%d],AppSize:[%d],Time:[%g], TimeUnits:[Time Per Iteration (ns)],NumIterations:[%u],RunType:[%d]\n",
      g_num_cores,
      g_num_elements,
      ((double) run_time_ns / (double) g_performed_iterations),
	   g_performed_iterations,
      g_run_type
      );
#endif

#ifdef DEBUG
  fprintf(stderr, "Done. Exiting...\n\n");
#endif

  return 0;
}

void setupExtentTable(long startmacsuffix, long startext, int numservers, size_t size)
{
   int fd;
   size_t nextents = (size - 1) / EXTENT_SZ + 1;
   size_t table_size = nextents * sizeof(uint64_t);
   volatile uint64_t *exttab;

   fd = open("/dev/dram-cache-exttab", O_RDWR);
   if (fd < 0) {
      perror("open exttab");
      abort();
   }

   if (table_size < PAGE_SZ)
      table_size = PAGE_SZ;

   exttab = mmap(NULL, table_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
   if (exttab == MAP_FAILED) {
      perror("mmap exttab");
      abort();
   }

   for (int i = 0; i < nextents; i++) {
      long server = i % numservers;
      uint64_t macsuffix = startmacsuffix + (server << 16);
      uint64_t extid = startext + (i / numservers);
      exttab[i] = (macsuffix << 48) | (1L << 47) | extid;
   }

   munmap(exttab, nextents * sizeof(uint64_t));
   close(fd);
}

int threadMain()
{
   uint64_t* arr_n_ptr;
   int fd;

   // pad out memory to next PAGE_SZ to make initialization easier...
   uint64_t num_total_elements_per_page = PAGE_SZ/sizeof(uint64_t);
   uint64_t num_elements_allocated = g_num_elements + (num_total_elements_per_page - (g_num_elements % num_total_elements_per_page));
   
   setupExtentTable(g_macaddr_suffix, g_start_extent, g_num_servers,
		   num_elements_allocated * sizeof(uint64_t));

   fd = open("/dev/dram-cache-mem", O_RDWR);
   if (fd < 0) {
      perror("open mem");
      abort();
   }
   arr_n_ptr = (uint64_t *) mmap(
	 NULL, num_elements_allocated * sizeof(uint64_t),
	 PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
   if (arr_n_ptr == MAP_FAILED) {
      perror("mmap memory");
      abort();
   }

   initializeGlobalArrays( arr_n_ptr,
                           g_num_elements,
                           g_stride);
   
   
   // clk_freq irrelevant 
   double const clk_freq = 0;
   g_performed_iterations = g_num_iterations;


   /** CRITICAL SECTION **/

//TODO time code ahead of time, and set num_iterations based on that...
#ifdef USE_MIN_TIME
   // run for g_num_iterations or until MIN_TIME reached, whichever comes last
   cctime_t volatile start_time = cc_get_seconds(clk_freq);
   cctime_t volatile estimated_end_time = start_time + MIN_TIME;
    
   intptr_t idx = 0;

   // we have to run for AT LEAST g_num_iterations, so dont muck up critical section 
   // with unnecessary code
   for (uint64_t k = 0; k < g_num_iterations; k++)
   {
      idx = arr_n_ptr[idx];
   }

   while (cc_get_seconds(clk_freq) < estimated_end_time)
   {
      g_performed_iterations += g_num_iterations;
      for (uint64_t k = 0; k < g_num_iterations; k++)
      {
         idx = arr_n_ptr[idx];
      }
   }

   cctime_t volatile stop_time = cc_get_seconds(clk_freq);

#else

   // run for g_num_iterations...
   cctime_t volatile start_time = cc_get_seconds(clk_freq);

   intptr_t idx = 0;

   for (uint64_t k = 0; k < g_num_iterations; k++)
   {
      idx = arr_n_ptr[idx];
   }

   cctime_t volatile stop_time = cc_get_seconds(clk_freq);

#endif 

   run_time_s = ((double) (stop_time - start_time)); 
   run_time_ns = run_time_s * 1.0E9;
   run_time_us = run_time_s * 1.0E6;
   run_time_ms = run_time_s * 1.0E3;

#ifdef DEBUG
   fprintf(stderr, "Total_Time (s)             : %g\n", run_time_s);
   fprintf(stderr, "Total_Time (ms)            : %g\n", run_time_ms);
   fprintf(stderr, "Total_Time (us)            : %g\n", run_time_us);
   fprintf(stderr, "Total_Time (ns)            : %g\n", run_time_ns);
#endif

   munmap(arr_n_ptr, num_elements_allocated * sizeof(uint64_t));
   close(fd);

   // prevent compiler from removing ptr chasing...
   // although the receiver must put idx into a volatile variable as well!
   return (int) idx; 
}


//generate a random linked list of a PAGE_SZ, starting at address arr_n_ptr[page_offset]
//the calling function is in charge of stitching the page together with the other pages
//num_accesses is the number of pointers we are going to add (not the number of
//elements spanned by the page)
//returns the index to the last index (to help the calling function stitch pages together)
//interleaving_space is the size between entries (i.e., 2 means generated lines are even/odd)
//assumes num_accesses can be divided by interleaving_space
uint64_t initializePage(uint64_t* arr_n_ptr, uint64_t page_offset, uint64_t num_accesses, uint64_t stride, uint64_t interleaving_space)
{
   if (interleaving_space > 2)
      printf("ERROR: unsupported interleaving_space\n");

#ifdef DEBUG
   printf("\n--(offset = %d) Generating Page-- num_el=%d, pageSz=%d, strd=%d\n\n",
      page_offset, num_accesses, num_accesses*stride*sizeof(int32_t), stride);
#endif

   cc_lfsr_t lfsr;
   uint64_t lfsr_init_val = 1; //TODO provide different streams different starting positions?
   //(-1) because we are going to loop through twice, once for odd and once for even entries 
   uint64_t lfsr_width = (log(num_accesses) / log(2)) - (interleaving_space - 1); //TODO EVENODD


   uint64_t max_accesses = (0x1 << lfsr_width)*interleaving_space;

//   printf("width=%d || max_accesses = %d, num_elements = %d\n", lfsr_width, max_accesses, num_accesses);

   // special case to handle non-powers of 2 num_accesses
   if (max_accesses < num_accesses)
      lfsr_width+=interleaving_space;


   uint64_t idx;
   uint64_t curr_idx, next_idx;

   curr_idx = page_offset;

//   printf("lfsr_width = %d\n", lfsr_width);

   // special cases to handle the VERY small pages (too small to use a LFSR)
   if (lfsr_width < 2)
   {
      // lfsr width too small, forget about randomizing this page...
      for (int i=0; i < num_accesses-1; i++)
      {
         arr_n_ptr[curr_idx] = curr_idx + stride;
         curr_idx += stride; 
      }
      
      arr_n_ptr[curr_idx] = page_offset;
      return curr_idx;
   }

   
   cc_lfsr_init(&lfsr, lfsr_init_val, lfsr_width);


   // "-1" because we do the last part separately (lfsr's don't generate 0s)
   for (int i=0; i < num_accesses-interleaving_space; i+=interleaving_space)
   {
//      printf("i=%d, i < %d-%d = %d\n", i, num_accesses, interleaving_space, (num_accesses-interleaving_space));
      //TODO do both even, odd at same time...
      next_idx = lfsr.value * (interleaving_space*stride) + page_offset; //TODO EVENODD
      
      arr_n_ptr[curr_idx] = next_idx;
      if (interleaving_space == 2)
         arr_n_ptr[curr_idx+stride] = next_idx+stride;


#ifdef DEBUG
      printf("   array[%4d] = %4d,   lfsr.value = %4d, i = %4d EVEN\n", 
                  curr_idx, arr_n_ptr[curr_idx], lfsr.value, i); 
      if (interleaving_space == 2)
         printf("   array[%4d] = %4d,   lfsr.value = %4d, i = %4d ODD\n", 
                     curr_idx+stride, arr_n_ptr[curr_idx+stride], lfsr.value, i); 
#endif

      curr_idx = next_idx;
      lfsr.value = cc_lfsr_next(&lfsr);

      //handle non powers of 2 num_accesses
//      while (lfsr.value > (num_accesses-1))
      while (lfsr.value > ((num_accesses-1)/interleaving_space))
      {
         lfsr.value = cc_lfsr_next(&lfsr);
      }
   }

   //handle last index, wrap back to the start of the page
   //let calling function stitch to next pages...
   if (interleaving_space==1)
   {
      arr_n_ptr[curr_idx] = page_offset;
   }
   else
   {
      arr_n_ptr[curr_idx] = page_offset+stride;
      arr_n_ptr[curr_idx+stride] = page_offset;
#ifdef DEBUG
      printf("  *array[%4d] = %4d,   lfsr.value = %4d, \n", 
                  curr_idx, arr_n_ptr[curr_idx], lfsr.value); 
      printf("  *array[%4d] = %4d,   lfsr.value = %4d, \n", 
                  curr_idx+stride, arr_n_ptr[curr_idx+stride], lfsr.value); 
#endif
      curr_idx += stride;
   }


   return curr_idx;
}


// this is the future initializeGlobalArrays, but it has one or two bugs so it's commented out for now
// it will more accurately provide the L3 access times on some processors
int initializeGlobalArrays(uint64_t* arr_n_ptr, uint64_t num_elements, uint64_t stride)
{
   // create a strided access array (no randomization)
   if(g_run_type != 0)
   {
      for (uint64_t i = 0; i < g_num_elements-1; i++)
      {
         arr_n_ptr[i % num_elements] = (i+stride) % num_elements;
      }
   }
   else
   {
      //randomize array on cacheline strided boundaries
#ifdef DEBUG
      printf("-==== Begin two-level randomization of the chase array ====-\n");
#endif
 
      // check to see if the array is 1-element long...
      if (num_elements == stride)
      {
#ifdef DEBUG
         printf("\nArray is 1 element long, returning....\n");
#endif
         // points back on itself
         arr_n_ptr[0] = 0;
         return 0;
      }
      
      uint64_t num_elements_per_page = PAGE_SZ/sizeof(uint64_t);

#ifdef DEBUG
      printf("\nnum_elements_per_page = %lu\n\n", PAGE_SZ/sizeof(uint64_t));
#endif
                    
      
      uint64_t last_idx;

      // two-level randomization (this is the outer level for-loop)
      // for each page...
      for(int i=0; i < num_elements; i+=num_elements_per_page)
      {
#ifdef DEBUG
         printf("\nStarting New Page: i=%d, page offset = %d\n", i, i); 
#endif
         
         uint64_t page_offset = i;


         //TODO test for partial page (which is always the last page)...
         if ((int32_t) i >= (int32_t) (num_elements - num_elements_per_page)  //is last page?
            && ((num_elements - i) % num_elements_per_page != 0))  //and last page is partial
         {
//            printf("PARTIAL PAGE: i=%d >= %d/%d - %d\n", i, num_elements, stride, num_elements_per_page);
            num_elements_per_page = num_elements - i; //num_elements % num_elements_per_page;
//            printf("PARTIAL PAGE: num_elements_per_page = %d,  total_num_elements = %d, i=%d\n",
//               num_elements_per_page,  num_elements, i);
         }



         last_idx = initializePage(arr_n_ptr, page_offset, num_elements_per_page/stride, stride, 1);
//         last_idx = initializePage(arr_n_ptr, page_offset, num_elements_per_page/stride, stride, 2);
         // tie this page to the next page...
         arr_n_ptr[last_idx] = (i+num_elements_per_page);
//         printf(" *array[%4d] = %4d, \n", last_idx, arr_n_ptr[last_idx]);
      }


      //handle the last page ...
      //wrap the array back to the start
      arr_n_ptr[last_idx] = 0;



   } //end of randomized array creation

#ifdef DEBUG
   printf("\n\n");
#endif

#ifdef PRINT_ARRAY
   printArray(arr_n_ptr, num_elements, stride);
#endif

   //verifyArray(arr_n_ptr, num_elements, stride);
  
   return 0;
}


int printArray(uint64_t *arr_ptr, uint64_t num_elements, uint64_t stride )
{
   uint64_t idx = 0;
   for (uint64_t i = 0; i < 2*num_elements/stride; i++)
   {
      fprintf(stderr, "%3d, ", arr_ptr[idx]);
      idx = arr_ptr[idx];
   }
   fprintf(stderr, "\n");
   
   fprintf(stderr, "arr_ptr: ");
   for (uint64_t i = 0; i < num_elements; i++)
   {
      if (i % stride == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%3d, ", arr_ptr[i]);
   }
   fprintf(stderr, "\n\n\n");

   return 0;
}


int verifyArray(uint64_t *arr_ptr, uint64_t num_elements, uint64_t stride)
{
   uint64_t idx = 0;
   uint64_t counter = 0;

   uint64_t first_idx = 0;
   uint64_t finished = 0;

   uint64_t* verify_array;
   verify_array = (uint64_t *) calloc(g_num_elements, sizeof(uint64_t));
   
   while (!finished)
   {
      verify_array[idx] = verify_array[idx] + 1;
      counter++;
      idx = arr_ptr[idx];

      if(counter >= num_elements || idx == first_idx)
         finished = 1;
   }

   uint64_t error = 0;

   for (uint64_t i = 0; i < num_elements; i+=stride)
   {
      if(verify_array[i] != 1)  
      {
#ifdef DEBUG
         fprintf(stderr, "Error at Element [%d], accessed %d times\n", 
            i,
            verify_array[i]
            );
#endif
         error = 1;
      }
   }

   if (counter==(num_elements/stride) && !error) {
#ifdef DEBUG
      fprintf(stderr,"Array verified\n\n");
#else
      int x=0; //here for compiler reasons
#endif
   } 
   else 
   {
      fprintf(stderr,"Error: Array size:%d, Stride size:%d, Loops in:%d (wants to loop in %d)\n\n",
         num_elements,
         stride,
         counter,
         num_elements/stride);
   }

   return error | (counter==(num_elements/stride));
}

