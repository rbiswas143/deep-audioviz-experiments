"""Send email notifications to configured recipients"""

import smtplib
from email.mime.text import MIMEText
import os
import json
import traceback

# Default email configuration
email_config = {
    'host': 'smtp.gmx.com',
    'port': 465,
    'username': None,
    'password': None,
    'recipients': []
}

# Override default configuration using a private config file
configured = False
email_config_file = 'private/email.json'
if os.path.isfile(email_config_file):
    with open(email_config_file, 'r') as cfg_file:
        email_config.update(json.load(cfg_file))
    configured = True
else:
    print('Warning: Emailer has not been configured. You can add a config file at {}'.format(email_config_file))


def sendmail(subject, body):
    """Create and send an email to the configured recipients"""
    if not configured:  # By default there is insufficient configuration to send a mail
        return
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = email_config['username']
    msg['To'] = email_config['recipients'][0]
    try:
        server = smtplib.SMTP_SSL(email_config['host'], email_config['port'])
        server.ehlo()
        server.login(email_config['username'], email_config['password'])
        server.sendmail(
            email_config['username'],
            email_config['recipients'],
            msg.as_string())
        server.quit()
        print('Mail sent')
    except:
        print('Error sending mail')
        traceback.print_exc()
