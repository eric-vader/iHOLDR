#!/usr/bin/env python3
import yaml
import hashlib
import json
import importlib
import os
import collections
import logging
import boto3

from config import Config

hash_args = lambda args: hashlib.sha1(json.dumps(args, sort_keys=True).encode()).hexdigest()

class Experiment:
    def __init__(self, exp_config):

        config = Config()

        # We guess if config is a file or not
        extension = os.path.splitext(exp_config)[-1]

        if extension == ".yml":
            if not os.path.isfile(exp_config):
                raise FileNotFoundError(f"Cannot experiment file at {exp_config}")
            logging.info(f"Load exp file from {exp_config}")
            self.exp_args = yaml.load(open(exp_config), yaml.FullLoader)
        else:
            endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')+":9000"
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

            assert(endpoint_url != None)
            assert(aws_access_key_id != None)
            assert(aws_secret_access_key != None)

            s3 = boto3.client('s3', 
                endpoint_url=endpoint_url, #'http://scarlettgpu1.d2.comp.nus.edu.sg:9000', 
                aws_access_key_id=aws_access_key_id, #'miniomlflow',
                aws_secret_access_key=aws_secret_access_key, #'R9RqzmC1',
                config=boto3.session.Config(signature_version='s3v4'),
                region_name='us-east-1')

            logging.info(f"Load exp file from server with key {exp_config}")
            obj = s3.get_object(Bucket="mlflow-jobs", Key=exp_config)
            self.exp_args = yaml.load(obj['Body'].read().decode('utf-8'), yaml.FullLoader)
        
        self.hash = hash_args(self.exp_args)
        # Simple checks

        self.sub_module_hashes = {}
        self.sub_module_name_lookup = {}
        self.sub_module_instances = {}
        
        # for module,v in self.exp_args.items():
        # Fix dependencies
        for module in ['datasets', 'algorithms']:
            v = self.exp_args[module]
            if type(v) != dict:
                raise Exception(f"Unexpected type for {module}, must be dict")
            if len(v.keys()) != 1:
                raise Exception(f"Must only init one object for every package")

            self.sub_module_hashes[module] = hash_args(v)
            (sub_module, kwargs), = v.items()
            self.sub_module_name_lookup[module] = sub_module

            Clazz = getattr(importlib.import_module(module), sub_module)
            instance = Clazz(hash=self.sub_module_hashes[module], **{**kwargs, **self.sub_module_instances})

            # if issubclass(Clazz, collections.abc.Callable):
            #     logging.info(f"Invoking object of {module}.{sub_module}: {instance}" )
            #     instance()
            
            self.sub_module_instances[module] = instance
