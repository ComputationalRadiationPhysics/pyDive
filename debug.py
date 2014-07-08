import sys
import time
from IPython.display import clear_output

def wait_watching_stdout(ar, dt=1, truncate=1000):
    while not ar.ready():
        stdouts = ar.stdout
        if not any(stdouts):
            continue
        # clear_output doesn't do much in terminal environments
        clear_output()
        print '-' * 30
        print "%.3fs elapsed" % ar.elapsed
        print ""
        for eid, stdout in zip(ar._targets, ar.stdout):
            if stdout:
                print "[ stdout %2i ]\n%s" % (eid, stdout[-truncate:])
        sys.stdout.flush()
        time.sleep(dt)
