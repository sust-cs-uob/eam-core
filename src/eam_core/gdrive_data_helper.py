from __future__ import print_function

from dateutil import parser
from tzlocal import get_localzone

local_timezone = get_localzone()
import pytz
import httplib2
from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

import logging

logging.basicConfig(level=logging.DEBUG)

import os
from datetime import datetime

# todo: untested. Does this matter?

SCOPES = 'https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/drive.readonly'
CLIENT_SECRET_FILE = 'gdrive_client_secret.json'
APPLICATION_NAME = 'Drive API Python Quickstart'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'drive-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    # if credentials are not existing -> login
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        credentials = tools.run_flow(flow, store, None)
        print('Storing credentials to ' + credential_path)
    return credentials


def get_service(credentials_json=None):
    if not credentials_json:
        credentials = get_credentials()
    else:
        credentials = client.Credentials.new_from_json(credentials_json)

    http = credentials.authorize(httplib2.Http())
    service = discovery.build('drive', 'v3', http=http, cache_discovery=False)

    return service


def get_most_recent_modifcation(service, file_id):
    r = service.revisions().list(fileId=file_id)
    data = r.execute()

    return parser.parse(data['revisions'][-1]['modifiedTime']), data['revisions'][-1]['id']


def download(service, file_id, outfile):
    """
    Downloads spreadsheet from google sheets as excel file.

    :param service:
    :type service:
    :param file_id:
    :type file_id:
    :param outfile:
    :type outfile:
    :return:
    :rtype:
    """
    path, _ = os.path.split(outfile)
    if not os.path.exists(path):
        os.makedirs(path)

    request = service.files().export_media(fileId=file_id,
                                           mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    with open(outfile, 'wb') as f:
        f.write(request.execute())


def update_local_data(file_id, local_file_path):
    """
    Compares the timestamp of the data file with that of the cloud file. Updates the local file if stale.
    :return:
    :rtype:
    """
    expanded_json = None

    # If environment variables are present, use these
    if "DATA_SHEET_ACCESS_TOKEN" in os.environ:
        logging.info("Creating credentials from environment variables")
        json_template = """{"access_token": "$DATA_SHEET_ACCESS_TOKEN",
            "client_id": "$DATA_SHEET_CLIENT_ID",
            "client_secret": "$DATA_SHEET_CLIENT_SECRET",
            "refresh_token": "$DATA_SHEET_REFRESH_TOKEN",
            "token_expiry": "2018-07-05T06:07:42Z",
            "token_uri": "https://accounts.google.com/o/oauth2/token",
            "user_agent": "Drive API Python Quickstart",
            "revoke_uri": "https://accounts.google.com/o/oauth2/revoke",
            "id_token": null,
            "id_token_jwt": null,
            "token_response": {
            "access_token": "$DATA_SHEET_ACCESS_TOKEN",
            "expires_in": 3600,
            "token_type": "Bearer"
            },
            "scopes": [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.file"
            ],
            "token_info_uri": "https://www.googleapis.com/oauth2/v3/tokeninfo",
            "invalid": false,
            "_class": "OAuth2Credentials",
            "_module": "oauth2client.client"
            }"""
        expanded_json = os.path.expandvars(json_template)

    service = get_service(credentials_json=expanded_json)

    modDTime, revision = get_most_recent_modifcation(service, file_id)

    try:
        mtime = os.path.getmtime(local_file_path)
    except OSError:
        mtime = 0
    last_modified_date = datetime.fromtimestamp(mtime, local_timezone)

    logging.debug(f'gdrive file revision time {modDTime}')
    logging.debug(f'local file mod time {last_modified_date.astimezone(pytz.utc)}')

    if modDTime > last_modified_date:
        logging.info(f'Remote file has been changed. Latest revision {revision}, updating local file {local_file_path}')
        download(service, file_id, local_file_path)

    logging.info('Local file is up-to-date')
    return modDTime, revision


if __name__ == '__main__':
    update_local_data()
