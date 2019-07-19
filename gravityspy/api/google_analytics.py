#!/usr/bin/env python

import argparse
import pandas
import numpy
import httplib2

from apiclient.discovery import build
from oauth2client import client
from oauth2client import file
from oauth2client import tools

SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
DISCOVERY_URI = ('https://analyticsreporting.googleapis.com/$discovery/rest')
CLIENT_SECRETS_PATH = 'client_secrets.json' # Path to client_secrets.json file.
VIEW_ID = '97262563'

def initialize_analyticsreporting():
    """Initializes the analyticsreporting service object.

    Returns:
    analytics an authorized analyticsreporting service object.
    """
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[tools.argparser])
    flags = parser.parse_args([])

    # Set up a Flow object to be used if we need to authenticate.
    flow = client.flow_from_clientsecrets(
        CLIENT_SECRETS_PATH, scope=SCOPES,
        message=tools.message_if_missing(CLIENT_SECRETS_PATH))

    # Prepare credentials, and authorize HTTP object with them.
    # If the credentials don't exist or are invalid run through the native client
    # flow. The Storage object will ensure that if successful the good
    # credentials will get written back to a file.
    storage = file.Storage('analyticsreporting.dat')
    credentials = storage.get()

    if credentials is None or credentials.invalid:
        credentials = tools.run_flow(flow, storage, flags)

    http = credentials.authorize(http=httplib2.Http())

    # Build the service object.
    analytics = build('analytics', 'v4', http=http, discoveryServiceUrl=DISCOVERY_URI)

    return analytics

def print_response(response):
    """Parses and prints the Analytics Reporting API V4 response"""

    for report in response.get('reports', []):
        columnHeader = report.get('columnHeader', {})
        dimensionHeaders = columnHeader.get('dimensions', [])
        metricHeaders = columnHeader.get('metricHeader', {}).get('metricHeaderEntries', [])
        rows = report.get('data', {}).get('rows', [])

        for row in rows:
            dimensions = row.get('dimensions', [])
            dateRangeValues = row.get('metrics', [])

            for header, dimension in zip(dimensionHeaders, dimensions):
                print(header + ': ' + dimension)

            for i, values in enumerate(dateRangeValues):
                print('Date range (' + str(i) + ')')
                for metricHeader, value in zip(metricHeaders, values.get('values')):
                    print(metricHeader.get('name') + ': ' + value)

def response_to_table(response):
    """Parses and makes a table from the Analytics Reporting API V4 response"""
    columns = response['reports'][0]['columnHeader']['dimensions']
    rows = numpy.vstack(pandas.DataFrame(response['reports'][0]['data']['rows'])['dimensions'].values)
    return pandas.DataFrame(rows,columns=columns)
  
# ---- Import standard modules to the python path.
def get_report(analytics, userID, subjectID=None, date_range=None, **kwargs):
    # Use the Analytics Service Object to query the Analytics Reporting API V4.
    verbose = kwargs.pop('verbose', False)
    dimensions = kwargs.pop('dimensions',  [{"name": "ga:eventAction"},
                        {"name": "ga:eventLabel"}, 
                         {"name": "ga:dimension4"}, 
                         {"name": 'ga:dateHourMinute'},
                        ])
    
    if date_range is not None:
        assert(type(date_range[0]) == 'str')
        assert(type(date_range[1]) == 'str')
        
    if date_range is None:
        date_range_passed = False
        date_range_dict = {"startDate": "7daysAgo", "endDate": "today"}
    else:
        date_range_passed = True
        try:
            # first try YYYYDDMM
            startDate = datetime.datetime.strptime(date_range[0], '%Y%m%d')
            endDate = datetime.datetime.strptime(date_range[1], '%Y%m%d')
            if verbose:
                print('You are requesting analytics data for a time period '
                      'beginning {0} and ending {1}'.format(startDate, endDate))
        except:
            # then try YYYYDDMMHHMM
            startDate = datetime.datetime.strptime(date_range[0], '%Y%m%d%H%M')
            endDate = datetime.datetime.strptime(date_range[1], '%Y%m%d%H%M')
            if verbose:
                print('You are requesting analytics data for a time period '
                      'beginning {0} and ending {1}'.format(startDate, endDate))
        finally:
            raise ValueError('Date must be in either YYYYDDMM or YYYYDDMMHHMM')
            
        startDate_string = startDate.strftime("%Y%m%d")
        endDate_string = endDate.strftime("%Y%m%d")
        dateHourMinuteStart_string = startDate.strftime('%Y%m%d%H%M')
        dateHourMinuteEnd_string = endDate.strftime('%Y%m%d%H%M')
        date_range_dict = {"startDate": "{0}".format(startDate_string), "endDate": "{}".format(endDate_string)}
        
    report_dict = {}
    # Date Range requested for report
    report_dict.update({'viewId': VIEW_ID})
    report_dict.update({'dateRanges': [date_range_dict]})
    report_dict.update({'metrics': [{'expression': 'ga:users'}]})
    report_dict.update({'dimensions': dimensions})
    
    if date_range_passed:
        dimensionFilterClauses = kwargs.pop('dimensionFilterClauses', [{
                                                                      "filters": [
                                                                          {
                                                                          "dimensionName": "ga:dimension3",
                                                                          "operator": "EXACT",
                                                                          "expressions": "zooniverse/gravity-spy"
                                                                          }],
                                                                      "filters": [
                                                                          {
                                                                          "dimensionName": "ga:dimension4",
                                                                          "operator": "EXACT",
                                                                          "expressions": "{0}".format(userID)
                                                                          }],
                                                                      "filters": [
                                                                          {
                                                                          "dimensionName": "ga:dateHourMinute",
                                                                          "operator": "NUMERIC_GREATER_THAN",
                                                                          "expressions": "{0}".format(dateHourMinuteStart_string)
                                                                          }],
                                                                      "filters": [
                                                                          {
                                                                          "dimensionName": "ga:dateHourMinute",
                                                                          "operator": "NUMERIC_LESS_THAN",
                                                                          "expressions": "{0}".format(dateHourMinuteEnd_string)
                                                                          }],
                                                                  }])
    else:
        dimensionFilterClauses = kwargs.pop('dimensionFilterClauses', [{
                                                                      "filters": [
                                                                          {
                                                                          "dimensionName": "ga:dimension3",
                                                                          "operator": "EXACT",
                                                                          "expressions": "zooniverse/gravity-spy"
                                                                          }],
                                                                      "filters": [
                                                                          {
                                                                          "dimensionName": "ga:dimension4",
                                                                          "operator": "EXACT",
                                                                          "expressions": "{0}".format(userID)
                                                                          }],
                                                                  }])

    report_dict.update({'dimensionFilterClauses': dimensionFilterClauses})
    
    return analytics.reports().batchGet(body={'reportRequests': [report_dict]}).execute()
