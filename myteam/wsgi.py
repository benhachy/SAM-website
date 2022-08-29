"""
WSGI config for myteam project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myteam.settings')

application = get_wsgi_application()


# ML registry
import inspect
from mlAPI.registry import MLRegistry
from mlAPI.Full.full import FullModel

try:
    registry = MLRegistry() # create ML registry
    # Random Forest classifier
    fm = FullModel()
    # add to ML registry
    registry.add_algorithm(endpoint_name="SAM",
                            algorithm_object=fm,
                            algorithm_name="SAM",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="MYTEAM",
                            algorithm_description="Synthesis Analysis Monitor",
                            algorithm_code=inspect.getsource(FullModel))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))

    