#ifndef __COMMON_H__
#define __COMMON_H__
static enum key_state {NOTPUSHED, PUSHED} keyarr[127];
static double floorDepth = -0.1;

static std::chrono::time_point<std::chrono::system_clock> time_check_s = std::chrono::system_clock::now();

static void time_check_start()
{
	time_check_s = std::chrono::system_clock::now();
}

static void time_check_end()
{
	std::chrono::duration<double> elapsed_seconds;
	elapsed_seconds = std::chrono::system_clock::now()-time_check_s;
	std::cout<<elapsed_seconds.count()<<std::endl;
}


#endif