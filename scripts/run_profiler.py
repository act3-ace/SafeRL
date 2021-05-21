import os 
import sys
import cProfile
import pstats


#out_file = sys.argv[1]
#cProfile.run('train.py',out_file)


#python -m cProfile [-o output_file] [-s sort_order] (-m module | myscript.py)



# param is the profile file
def interpret(profile_file,results_file):
    with open(results_file,'w') as stream:  
        p = pstats.Stats(profile_file,stream=stream)
        p.sort_stats('cumulative').print_stats(20) 

results_file = 'docking_prof_results_cumul_time.txt'
out_file = 'docking_multi_profile_results.txt' 
interpret(out_file,results_file)

