'''
Author: 21308004-Yao Yuan
Date: 2023-5-20
'''

"""
ASGI config for first_try project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "first_try.settings")

application = get_asgi_application()
