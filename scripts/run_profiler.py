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
        p.print_stats()
        #p.sort_stats('cumulative').print_stats(20) 

results_file = 'env_step_profile_results'
out_file = 'env_step_profile' 
interpret(out_file,results_file)

