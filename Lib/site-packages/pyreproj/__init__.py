# -*- coding: utf-8 -*-
import requests


from pyproj import Proj, transform as pyproj_transform
from functools import partial
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from pyreproj.exception import InvalidFormatError, InvalidTargetError


__version__ = '2.0.0'


class Reprojector(object):
    """
    Creates a new Reprojector instance.

    :param srs_service_url: The url of the service to be used by get_projection_from_service.
        It can contain a parameter "{epsg}"
    :type srs_service_url: :obj:`str`, optional
    """

    def __init__(self, srs_service_url='http://spatialreference.org/ref/epsg/{epsg}/proj4/'):
        self.srs_service_url = srs_service_url

    def get_transformation_function(self, from_srs=4326, to_srs=4326):
        """
        This method creates a transformation function to transform coordinates from one reference system to
        another one. The projections are set using the two parameters which can match one of the following
        types each:

        - **int:** The EPSG code as integer
        - **str:** The proj4 definition string or the EPSG code as string including the "epsg:" prefix.
        - **pyproj.Proj:** An instance of :class:`pyproj.Proj`, for example as returned by
          :func:`~pyreproj.Reprojector.get_projection_from_service`.

        :param from_srs: Spatial reference system to transform from. Defaults to 4326.
        :type from_srs: :obj:`int`, :obj:`str`, :obj:`pyproj.Proj`
        :param to_srs: Spatial reference system to transform to. Defaults to 4326.
        :type to_srs: :obj:`int`, :obj:`str`, :obj:`pyproj.Proj`
        :return: A function accepting two arguments, x and y.
        :rtype: :func:`functools.partial`
        """
        return partial(
            pyproj_transform,
            self._get_proj_from_parameter(from_srs),
            self._get_proj_from_parameter(to_srs)
        )

    def get_projection_from_service(self, epsg=4326):
        """
        Sends a request to the set service with the given EPSG code and create an instance of
        :class:`pyproj.Proj` from the received definition.

        :param epsg: The EPSG code for the projection.
        :type epsg: :obj:`int`
        :return: The resulting projection for the received definition.
        :rtype: :obj:`pyproj.Proj`
        """
        url = self.srs_service_url.format(epsg=epsg)
        response = requests.get(url)
        if response.status_code == requests.codes.ok:
            return Proj(response.text)
        response.raise_for_status()

    def transform(self, target, from_srs=4326, to_srs=4326):
        """
        This method takes a geometry or coordinate and transforms it from the source to the target reference
        system. The projections can be defined the same way as for
        :func:`~pyreproj.Reprojector.get_transformation_function`.

        :param target: The object to be transformed.
        :type target: :obj:`list`, :obj:`tuple`, :obj:`shapely.geometry.base.BaseGeometry`
        :param from_srs: Spatial reference system to transform from. Defaults to 4326.
        :type from_srs: :obj:`int`, :obj:`str`, :obj:`pyproj.Proj`
        :param to_srs: Spatial reference system to transform to. Defaults to 4326.
        :type to_srs: :obj:`int`, :obj:`str`, :obj:`pyproj.Proj`
        :return: The transformed object.
        :rtype: :obj:`list`, :obj:`tuple`, :obj:`shapely.geometry.base.BaseGeometry`
        :raise: :class:`~pyreproj.exception.InvalidFormatError`,
            :class:`~pyreproj.exception.InvalidTargetError`
        """
        from_proj = self._get_proj_from_parameter(from_srs)
        to_proj = self._get_proj_from_parameter(to_srs)
        if isinstance(target, list) and len(target) > 1:
            return self._transform_list(target, from_proj, to_proj)
        elif isinstance(target, tuple):
            return self._transform_tuple(target, from_proj, to_proj)
        elif isinstance(target, BaseGeometry):
            return self._transform_shapely(target, from_proj, to_proj)
        else:
            msg = 'Invalid target to transform. Valid targets are [x, y], (x, y) or a shapely geometry ' \
                  'object.'
            raise InvalidTargetError(msg)

    @staticmethod
    def _transform_list(target, from_proj, to_proj):
        x, y = pyproj_transform(from_proj, to_proj, target[0], target[1])
        return [x, y]

    @staticmethod
    def _transform_tuple(target, from_proj, to_proj):
        x, y = pyproj_transform(from_proj, to_proj, target[0], target[1])
        return x, y

    def _transform_shapely(self, target, from_proj, to_proj):
        project = self.get_transformation_function(from_proj, to_proj)
        return shapely_transform(project, target)

    @staticmethod
    def _get_proj_from_parameter(param):
        msg = 'Invalid projection definition. Valid formats are "pyproj.Proj object", EPSG code as integer' \
              ' or string with leading "epsg:" or a proj4 definition string'
        if isinstance(param, Proj):
            proj = param
        elif isinstance(param, str) and param.lower().startswith('epsg:'):
            proj = Proj(projparams=int(param[5:]))
        else:
            try:
                proj = Proj(projparams=param)
            except RuntimeError:
                raise InvalidFormatError(msg)
        return proj
