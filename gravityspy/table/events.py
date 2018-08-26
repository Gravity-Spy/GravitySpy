# -*- coding: utf-8 -*-
# Copyright (C) Scott Coughlin (2017-)
#
# This file is part of gravityspy.
#
# gravityspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gravityspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gravityspy.  If not, see <http://www.gnu.org/licenses/>.

from gwtrigfind import find_trigger_files
from gwpy.segments import DataQualityFlag
from gwpy.table import GravitySpyTable

from ..utils import log
from ..utils import utils
from ..api.project import GravitySpyProject
from ..ml.train_classifier import make_model

import panoptes_client
import numpy
import pandas
import subprocess
import string
import random
import os

class Events(GravitySpyTable):
    """This class provides method for classifying omicron triggers with gravityspy
    """
    @classmethod
    def read(cls, *args, **kwargs):
        """Classify triggers in this table

        Parameters:
        ----------

        Returns
        -------
        """
        etg = kwargs.pop('etg', 'OMICRON')
        tab = super(Events, cls).read(*args, **kwargs)
        tab = tab.to_pandas()
        if 'gravityspy_id' not in tab.columns:
            tab['gravityspy_id'] = tab.apply(id_generator, 1)
            tab['image_status'] = 'testing'
            tab['data_quality'] = 'no_flag'
            tab['upload_flag'] = 0
            tab['citizen_score'] = 0.0
            tab['links_subjects'] = 0
            tab['url1'] = ''
            tab['url2'] = ''
            tab['url3'] = ''
            tab['url4'] = ''

        if etg == 'OMICRON':
            tab['event_id'] = tab['event_id'].apply(int)
            tab['process_id'] = tab['process_id'].apply(int)

        tab = cls.from_pandas(tab)

        if etg == 'OMICRON':
            tab['event_time'] = tab['peak_time'] + (0.000000001)*tab['peak_time_ns']
            tab['event_time'].format = '%.9f'
        else:
            raise ValueError('No trigger reading has been defined for this ETG')

        return tab
        
    def classify(self, project_info_pickle, path_to_cnn, **kwargs):
        """Classify triggers in this table

        Parameters:
        ----------

        Returns
        -------
        """
        if 'event_time' not in self.keys():
            raise ValueError("This method only works if you have defined "
                             "a column event_time for your Event Trigger Generator.")

        config = kwargs.pop('config', utils.GravitySpyConfigFile())

        # Parse Ini File
        plot_time_ranges = config.plot_time_ranges
        plot_normalized_energy_range = config.plot_normalized_energy_range

        plot_directory = kwargs.pop('plot_directory', 'plots')

        for event_time, ifo, channel, gid in zip(self['event_time'], self['ifo'],
                                                 self['channel'], self['gravityspy_id']):
            specsgrams, q_value = utils.make_q_scans(event_time=event_time, config=config,
                                                     **kwargs) 

            utils.save_q_scans(plot_directory, specsgrams,
                               plot_normalized_energy_range, plot_time_ranges,
                               ifo, event_time, id_string=gid,
                               **kwargs)

        self['q_value'] = q_value

        results = utils.label_q_scans(plot_directory=plot_directory,
                                      path_to_cnn=path_to_cnn,
                                      project_info_pickle=project_info_pickle,
                                      **kwargs)

        results = results.to_pandas()
        results['Filename1'] = results['Filename1'].apply(lambda x, y : os.path.join(y, x),
                                                          args=(plot_directory,))
        results['Filename2'] = results['Filename2'].apply(lambda x, y : os.path.join(y, x),
                                                          args=(plot_directory,))
        results['Filename3'] = results['Filename3'].apply(lambda x, y : os.path.join(y, x),
                                                          args=(plot_directory,))
        results['Filename4'] = results['Filename4'].apply(lambda x, y : os.path.join(y, x),
                                                          args=(plot_directory,))

        
        results = Events.from_pandas(results.merge(self.to_pandas(),
                                                   on=['gravityspy_id']))

        return results

    def to_sql(self, table='glitches_v2d0', engine=None, **kwargs):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
        ----------

        Returns
        -------
        """
        from sqlalchemy.engine import create_engine
        # connect if needed
        if engine is None:
            conn_kw = {}
            for key in ('db', 'host', 'user', 'passwd'):
                try:
                    conn_kw[key] = kwargs.pop(key)
                except KeyError:
                    pass
            engine = create_engine(get_connection_str(**conn_kw))

        self.to_pandas().to_sql(table, engine, index=False, if_exists='append')
        return

    def update_sql(self, table='glitches_v2d0', engine=None):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
        ----------

        Returns
        -------
        """
        from sqlalchemy.engine import create_engine
        # connect if needed
        if engine is None:
            conn_kw = {}
            for key in ('db', 'host', 'user', 'passwd'):
                try:
                    conn_kw[key] = kwargs.pop(key)
                except KeyError:
                    pass
            engine = create_engine(get_connection_str(**conn_kw))

        column_dict = self.to_pandas().to_dict(orient='records')[0]
        sql_command = 'UPDATE {0} SET '.format(table)
        for column_name in column_dict:
            if isinstance(column_dict[column_name], str):
                sql_command = sql_command + '''\"{0}\" = \'{1}\', '''.format(column_name, column_dict[column_name])
            else:
                sql_command = sql_command + '''\"{0}\" = {1}, '''.format(column_name, column_dict[column_name])
        sql_command = sql_command[:-2] + ' WHERE \"gravityspy_id\" = \'' + self['gravityspy_id'].iloc[0] + "'"
        engine.execute(sql_command)
        return

    def upload_to_zooniverse(self, subject_set_id=None):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
        ----------

        Returns
        -------
        """
        # First filter out images that have already been uploaded
        tab = self[self['upload_flag'] == 1]

        if subject_set_id is None:
            subset_ids = numpy.unique(tab['subjectset'])
        else:
            subset_ids = numpy.atleast_1d(numpy.array(subject_set_id))

        panoptes_client.Panoptes.connect()
        project = panoptes_client.Project.find(slug='zooniverse/gravity-spy')

        for subset_id in subset_ids:
            subjectset = panoptes_client.SubjectSet.find(subset_id)
            subjects = []

            if subject_set_id is None:
                tab1 = tab[tab['subjectset'] == subset_id]

            for fn1, fn2, fn3, fn4, gid in tab1['Filename1', 'Filename2', 'Filename3', 'Filename4', 'gravityspy_id']:
                subject = panoptes_client.Subject()
                subject.links.project = project
                subject.add_location(str(fn1))
                subject.add_location(str(fn2))
                subject.add_location(str(fn3))
                subject.add_location(str(fn4))
                subject.metadata['date'] = '20180825'
                subject.metadata['subject_id'] = str(gid)
                subject.metadata['Filename1'] = fn1.split('/')[-1]
                subject.metadata['Filename2'] = fn2.split('/')[-1]
                subject.metadata['Filename3'] = fn3.split('/')[-1]
                subject.metadata['Filename4'] = fn4.split('/')[-1]
                subject.save()
                subjects.append(subject)
                self[self['gravityspy_id'] == gid]['links_subject'] = int(subject.id)
                self[self['gravityspy_id'] == gid]['url1'] = subject.raw['locations'][0]['image/png'].split('?')[0]
                self[self['gravityspy_id'] == gid]['url2'] = subject.raw['locations'][1]['image/png'].split('?')[0]
                self[self['gravityspy_id'] == gid]['url3'] = subject.raw['locations'][2]['image/png'].split('?')[0]
                self[self['gravityspy_id'] == gid]['url4'] = subject.raw['locations'][3]['image/png'].split('?')[0]
                self['upload_flag'][self['gravityspy_id'] == gid] = 1
            subjectset.add(subjects)

        return self

    def update_scores(self, path_to_cnn):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
        ----------

        Returns
        -------
        """
        if ['Filename1', 'Filename2', 'Filename3', 'Filename4'] not in self.keys():
            raise ValueError("This method only works if the file paths of the images "
                             "of the images are known.")

        return

    def determine_workflow_and_subjectset(self, project_info_pickle):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
        ----------

        Returns
        -------
        """
        if 'ml_confidence' not in self.keys() or 'ml_label' not in self.keys():
            raise ValueError("This method only works if the confidence and label "
                             "of the image in known.")
        gspyproject = GravitySpyProject.load_project_from_cache(
                                                                project_info_pickle
                                                                )

        workflows_for_each_class = gspyproject.get_level_structure(IDfilter='O2')
        # Determine subject set and workflow this should go to.
        level_of_images = []
        subjectset_of_images = []
        for label, confidence in zip(self['ml_label'], self['ml_confidence']):
            for iworkflow in ['1610', '1934', '1935', '2360', '2117']:
                if label in workflows_for_each_class[iworkflow].keys():
                     if workflows_for_each_class[iworkflow][label][2][1] <= \
                            confidence <= \
                                 workflows_for_each_class[iworkflow][label][2][0]:
                         level_of_images.append(int(workflows_for_each_class[iworkflow][label][0]))
                         subjectset_of_images.append(workflows_for_each_class[iworkflow][label][1])
                         break

        self["workflow"] = level_of_images
        self["subjectset"] = subjectset_of_images

        return self

    @classmethod
    def get_triggers(cls, start, end, channel,
                     dqflag, verbose=True, **kwargs):
        """Obtain omicron triggers to run gravityspy on

        Parameters:
        ----------

        Returns
        -------
        """
        duration_max = kwargs.pop('duration_max', None)
        duration_min = kwargs.pop('duration_min', None)
        frequency_max = kwargs.pop('frequency_max', 2048)
        frequency_min = kwargs.pop('frequency_min', 10)
        snr_max = kwargs.pop('snr_max', None)
        snr_min = kwargs.pop('snr_min', 7.5)

        detector = channel.split(':')[0]

        logger = log.Logger('Gravity Spy: Fetching Omicron Triggers')

        # Obtain segments that are analysis ready
        analysis_ready = DataQualityFlag.query('{0}:{1}'.format(detector, dqflag),\
                                              float(start), float(end))

        # Display segments for which this flag is true
        logger.info("Segments for which the {0} Flag "
                    "is active: {1}".format(dqflag, analysis_ready.active))

        # get Omicron triggers
        files = find_trigger_files(channel,'Omicron',
                                   float(start),float(end))

        omicrontriggers = cls.read(files, tablename='sngl_burst', format='ligolw')

        masks = numpy.ones(len(omicrontriggers), dtype=bool)

        if not duration_max is None:
            masks &= (omicrontriggers['duration'] <= duration_max)
        if not duration_min is None:
            masks &= (omicrontriggers['duration'] >= duration_min)
        if not frequency_max is None:
            masks &= (omicrontriggers['peak_frequency'] <= frequency_max)
        if not frequency_min is None:
            masks &= (omicrontriggers['peak_frequency'] >= frequency_min)
        if not snr_max is None:
            masks &= (omicrontriggers['snr'] <= snr_max)
        if not snr_min is None:
            masks &= (omicrontriggers['snr'] >= snr_min)

        omicrontriggers = omicrontriggers[masks]
        # Set peakGPS
        omicrontriggers['peakGPS'] = omicrontriggers['peak_time'] + (0.000000001)*omicrontriggers['peak_time_ns']

        logger.info("List of available metadata information for a given glitch provided by omicron: {0}".format(omicrontriggers.keys()))

        logger.info("Number of triggers after SNR and Freq cuts but before ANALYSIS READY flag filtering: {0}".format(len(omicrontriggers)))

        # Filter the raw omicron triggers against the ANALYSIS READY flag.
        vetoed = omicrontriggers['peakGPS'].in_segmentlist(analysis_ready.active)
        omicrontriggers = omicrontriggers[vetoed]

        logger.info("Final trigger length: {0}".format(len(omicrontriggers)))

        return omicrontriggers


def id_generator(x, size=10,
                 chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    """Obtain omicron triggers run gravityspy on

    Parameters:
    ----------

    Returns
    -------
    """
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def get_connection_str(db='gravityspy',
                       host='gravityspy.ciera.northwestern.edu',
                       user=os.getenv('GRAVITYSPY_DATABASE_USER', None),
                       passwd=os.getenv('GRAVITYSPY_DATABASE_PASSWD', None)):
    """Create string to pass to create_engine
    """
    if (not user) or (not passwd):
        raise ValueError('Remember to either pass '
                         'or export GRAVITYSPY_DATABASE_USER '
                         'and export GRAVITYSPY_DATABASE_PASSWD in order '
                         'to access the Gravity Spy Data: '
                         'https://secrets.ligo.org/secrets/144/'
                         ' description is username and secret is password.')

    return 'postgresql://{0}:{1}@{2}:5432/{3}'.format(user, passwd, host, db)
