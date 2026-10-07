"""Planner adapters.

``base.NavsimPlanner`` runs any native navsim ``AbstractAgent`` from an agent config alone;
an agent config names a subclass through ``model.adapter`` (``file.py:Class`` or
``package.module:Class``) only when the model needs one. ``recogdrive`` and
``recogdrive_sdroute`` are the shipped examples of such subclasses.
"""
