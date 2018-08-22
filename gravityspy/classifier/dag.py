from gwpy.table import EventTable
class GravitySpyDag(EventTable):
    """This class provided help methods for submitting to condor
    """
    def write_dag():
        with open('gravityspy_{0}_{1}.dag'.format(oTriggers.peak_time.min(),oTriggers.peak_time.max()),'a+') as dagfile:
            dagfile.write('JOB {0}{1}{2} ./condor/{3}\n'.format(x.peak_time,x.peak_time_ns,x.event_id, opts.subfile))
            dagfile.write('RETRY {0}{1}{2} 10\n'.format(x.peak_time,x.peak_time_ns,x.event_id))
            dagfile.write('VARS {0}{1}{4} jobNumber="{0}{1}{4}" eventTime="{2}" ID="{3}"'.format(x.peak_time,x.peak_time_ns,x.peakGPSforDag,x.uniqueID,x.event_id))
        return
    def write_sub():
        return
