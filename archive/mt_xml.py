# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:33:12 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
from pathlib import Path
from configparser import ConfigParser
from archive.utils import xml_utils
import datetime


class MTSBXML(xml_utils.XMLRecord):
    """
    Class to make it easier to create and manipulate Sciencebase metadata XML files
    that adhere to the FGDC standards.

    Based on pymdwizard https://github.com/usgs/fort-pymdwizard/tree/master

    """

    def __init__(self, fn=None):
        super().__init__(fn)

    def _get_date(self, date):
        return xml_utils.format_date(date)

    def read_template_xml(self, template_fn):
        """
        Read in a template xml that can be updated later

        :param template_fn: Full path to template XML file
        :type template_fn: string or Path
        :return: fills
        :rtype: TYPE

        """

        if not Path(template_fn):
            raise IOError(f"Could not find {template_fn}")

        self.fn = template_fn

    def read_config(self, cfg_fn):
        """
        read from a config file
        """
        cfg_fn = Path(cfg_fn)
        if not cfg_fn.exists():
            raise IOError(f"Could not find {cfg_fn}")
        c = ConfigParser()
        c.read_file(cfg_fn.open())
        return c

    def update_from_config(self, cfg_fn, child=False):

        c = self.read_config(cfg_fn)

        self._run_update(self._update_cite_info, c["general"])
        try:
            self._run_update(self._update_journal_citation, c["journal_citation"])
        except KeyError:
            print("WARNING: No journal information skipping")
        self._run_update(self._update_description, c["general"])
        for k in ["general", "thesaurus"]:
            self._run_update(self._update_keywords, c["keywords"][k], k)
        for k in ["common", "gnis", "terranes"]:
            self._run_update(self._update_places, c["places"][k], k)
        for k in ["geolex", "eras"]:
            self._run_update(self._update_temporal, c["temporal"][k], k)

        self._run_update(self._update_constraints, c["usage"]["constraints"])
        self._run_update(self._update_contact, c["principle_investigator"])
        self._run_update(self._update_processing, c["processing"])
        if not child:
            self._run_update(self._update_attachments, c["attachments"])

        self._run_update(self.update_metadate)

    @staticmethod
    def _run_update(function, *args):
        try:
            function(*args)
        except KeyError as error:
            print(f"Cound not run {function.__func__.__name__} with {args}")
            print("Error:")
            print(error)

    def _update_authors(self, authors, orcids=None):
        authors = [name.strip() for name in authors.split(",")]
        self.metadata.idinfo.citation.citeinfo.clear_children("origin")
        count = 0
        other_cite = ""
        for name in authors:
            xml_utils.XMLNode(
                tag="origin",
                text=name.strip(),
                parent_node=self.metadata.idinfo.citation.citeinfo,
                index=count,
            )
            count += 1
        if orcids:
            orcids = [number.strip() for number in orcids.split(",")]

            other_cite = "Additional information about Originators: "
            for name, orcid in zip(authors, orcids):
                other_cite += f"{name}, {orcid}; "

        return other_cite

    def _update_cite_info(self, config_object):
        """
        Update Cite information from config file information
        """

        mapping = {
            "authors": "origin",
            "title": "title",
            "release_date": "pubdate",
            "doi": "onlink",
            "gis_data": "geoform",
            "suggested_citation": "othercit",
        }
        other_cite = ""
        for k in [
            "authors",
            "release_date",
            "gis_data",
            "title",
            "suggested_citation",
            "doi",
        ]:
            v = config_object[k]
            xml_key = mapping[k]
            if k == "authors":
                if "orcids" in config_object.keys():
                    other_cite = self._update_authors(v, config_object["orcids"])
                else:
                    other_cite = self._update_authors(v)

            elif k == "suggested_citation":
                other_cite += f"\nSuggested citation: {v}"
                self.metadata.idinfo.citation.citeinfo.othercit.text = other_cite

            elif k == "release_date":
                self.metadata.idinfo.citation.citeinfo.pubdate.text = self._get_date(v)

            else:
                setattr(
                    getattr(self.metadata.idinfo.citation.citeinfo, xml_key), "text", v
                )

    def _update_journal_citation(self, config_object):
        """
        Add a journal citation
        """

        lworkcit = xml_utils.XMLNode(
            tag="lworkcit", parent_node=self.metadata.idinfo.citation.citeinfo
        )

        citation = xml_utils.XMLNode(tag="citeinfo", parent_node=lworkcit)

        count = 0
        for name in config_object["authors"].split(","):
            xml_utils.XMLNode(
                tag="origin", text=name.strip(), parent_node=citation, index=count
            )
            count += 1

        xml_utils.XMLNode(
            tag="pubdate",
            text=self._get_date(config_object["date"]),
            parent_node=citation,
        )

        xml_utils.XMLNode(
            tag="title", text=config_object["title"], parent_node=citation
        )

        xml_utils.XMLNode(tag="geoform", text="Publication", parent_node=citation)

        serinfo = xml_utils.XMLNode(tag="serinfo", parent_node=citation)
        xml_utils.XMLNode(
            tag="sername", text=config_object["journal"], parent_node=serinfo
        )
        xml_utils.XMLNode(tag="issue", text=config_object["issue"], parent_node=serinfo)

        xml_utils.XMLNode(tag="onlink", text=config_object["doi"], parent_node=citation)

    def _update_description(self, config_object):
        """
        update description
        """
        if "purpose" in config_object.keys():
            self.metadata.idinfo.descript.purpose.text = config_object["purpose"]

        if "abstract" in config_object.keys():
            self.metadata.idinfo.descript.abstract.text = config_object["abstract"]

        if "supplement_info" in config_object.keys():
            self.metadata.idinfo.descript.supplinfo.text = config_object[
                "supplement_info"
            ]

    def update_bounding_box(self, west, east, north, south):
        """
        update the bounding box
        """
        self.metadata.idinfo.spdom.bounding.westbc.text = f"{float(west):.5f}"
        self.metadata.idinfo.spdom.bounding.eastbc.text = f"{float(east):.5f}"
        self.metadata.idinfo.spdom.bounding.northbc.text = f"{float(north):.5f}"
        self.metadata.idinfo.spdom.bounding.southbc.text = f"{float(south):.5f}"

    def _update_keywords(self, kw_string, key):
        """
        update the keywords
        """
        index_dict = {"general": 1, "thesaurus": 2}

        kw = [k.strip() for k in kw_string.split(",")]
        try:
            count = len(self.metadata.idinfo.keywords.theme[index_dict[key]].themekey)
        except TypeError:
            count = 1
        self.metadata.idinfo.keywords.theme[index_dict[key]].clear_children("themekey")
        for k in kw:
            xml_utils.XMLNode(
                tag="themekey",
                text=k,
                parent_node=self.metadata.idinfo.keywords.theme[index_dict[key]],
                index=count,
            )
            count += 1

    def _update_places(self, place_string, place):
        """
        update places
        """
        index_dict = {"gnis": 0, "common": 1, "terranes": 2}

        index = index_dict[place]
        kw = [k.strip() for k in place_string.split(",")]
        try:
            count = len(self.metadata.idinfo.keywords.place[index].placekey)
        except TypeError:
            count = 1
        self.metadata.idinfo.keywords.place[index].clear_children("placekey")
        for k in kw:
            xml_utils.XMLNode(
                tag="placekey",
                text=k,
                parent_node=self.metadata.idinfo.keywords.place[index],
                index=count,
            )
            count += 1

    def _update_temporal(self, temporal_string, temporal):
        """
        update places
        """
        index_dict = {"geolex": 0, "eras": 1}

        index = index_dict[temporal]
        kw = [k.strip() for k in temporal_string.split(",")]
        try:
            count = len(self.metadata.idinfo.keywords.temporal[index].tempkey)
        except TypeError:
            count = 1
        self.metadata.idinfo.keywords.temporal[index].clear_children("tempkey")
        for k in kw:
            xml_utils.XMLNode(
                tag="tempkey",
                text=k,
                parent_node=self.metadata.idinfo.keywords.temporal[index],
                index=count,
            )
            count += 1

    def _update_constraints(self, constraint_string):
        """

        :param constraint_string: DESCRIPTION
        :type constraint_string: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        self.metadata.idinfo.useconst.text = constraint_string

    def _update_contact(self, config_object):
        """
        Update ptcontact and metc

        :param config_object: DESCRIPTION
        :type config_object: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        self.metadata.idinfo.ptcontac.cntinfo.cntperp.cntper.text = config_object[
            "name"
        ]
        self.metadata.metainfo.metc.cntinfo.cntperp.cntper.text = config_object["name"]

        self.metadata.idinfo.ptcontac.cntinfo.cntperp.cntorg.text = config_object[
            "orginization"
        ]
        self.metadata.metainfo.metc.cntinfo.cntperp.cntorg.text = config_object[
            "orginization"
        ]

        self.metadata.idinfo.ptcontac.cntinfo.cntpos.text = config_object["position"]
        self.metadata.metainfo.metc.cntinfo.cntpos.text = config_object["position"]

        self.metadata.idinfo.ptcontac.cntinfo.cntaddr.addrtype.text = "mailing address"
        self.metadata.metainfo.metc.cntinfo.cntaddr.addrtype.text = "mailing address"

        self.metadata.idinfo.ptcontac.cntinfo.cntaddr.address.text = config_object[
            "address"
        ]
        self.metadata.metainfo.metc.cntinfo.cntaddr.address.text = config_object[
            "address"
        ]

        self.metadata.idinfo.ptcontac.cntinfo.cntaddr.city.text = config_object["city"]
        self.metadata.metainfo.metc.cntinfo.cntaddr.city.text = config_object["city"]

        self.metadata.idinfo.ptcontac.cntinfo.cntaddr.state.text = config_object[
            "state"
        ]
        self.metadata.metainfo.metc.cntinfo.cntaddr.state.text = config_object["state"]

        self.metadata.idinfo.ptcontac.cntinfo.cntaddr.postal.text = config_object[
            "postal"
        ]
        self.metadata.metainfo.metc.cntinfo.cntaddr.postal.text = config_object[
            "postal"
        ]

        self.metadata.idinfo.ptcontac.cntinfo.cntaddr.country.text = config_object[
            "country"
        ]
        self.metadata.metainfo.metc.cntinfo.cntaddr.country.text = config_object[
            "country"
        ]

        self.metadata.idinfo.ptcontac.cntinfo.cntvoice.text = config_object["phone"]
        self.metadata.metainfo.metc.cntinfo.cntvoice.text = config_object["phone"]

        # self.metadata.idinfo.ptcontac.cntinfo.cntfax.text = config_object["fax"]
        # self.metadata.metainfo.metc.cntinfo.cntfax.text = config_object["fax"]

        self.metadata.idinfo.ptcontac.cntinfo.cntemail.text = config_object["email"]
        self.metadata.metainfo.metc.cntinfo.cntemail[0].text = config_object["email"]
        self.metadata.metainfo.metc.cntinfo.cntemail[1].text = config_object[
            "data_managment_email"
        ]

        self.metadata.idinfo.datacred.text = config_object["funding_source"]

    def _update_processing(self, config_object):
        """
        Update processing steps
        :param config_object: DESCRIPTION
        :type config_object: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.metadata.dataqual.lineage.clear_children("procstep")
        for step in [k for k in config_object.keys() if "step" in k]:
            step_count = int(step[-2:])
            index = step_count + 1
            date = config_object[f"date_{step_count:02}"].replace("-", "")

            xml_utils.XMLNode(
                tag="procstep",
                parent_node=self.metadata.dataqual.lineage,
                index=index - 1,
            )

            xml_utils.XMLNode(
                tag="procdesc",
                text=config_object[step],
                parent_node=self.metadata.dataqual.lineage.procstep[index + 1],
                index=0,
            )

            xml_utils.XMLNode(
                tag="procdate",
                text=date,
                parent_node=self.metadata.dataqual.lineage.procstep[index + 1],
                index=1,
            )

    def _update_attachments(self, config_object):
        """
        Update processing steps
        :param config_object: DESCRIPTION
        :type config_object: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.metadata.eainfo.clear_children("overview")
        try:
            count = len(self.metadata.eainfo.detailed.attr)
        except AttributeError:
            count = 0

        for ii, ext in enumerate(["edi", "png", "guide"]):
            overview = xml_utils.XMLNode(
                tag="overview", parent_node=self.metadata.eainfo, index=ii + count
            )

            xml_utils.XMLNode(
                tag="eaover",
                text=config_object[f"{ext}.fn"],
                parent_node=overview,
                index=0,
            )

            xml_utils.XMLNode(
                tag="eadetcit",
                text=config_object[f"{ext}.description"],
                parent_node=overview,
                index=1,
            )

    def update_dates(self, date=None):
        """
        Update release and metadata dates to input date or if not input the current
        date

        :param date: DESCRIPTION, defaults to None
        :type date: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if date is None:
            date = datetime.datetime.utcnow()
            date = date.strftime("%Y%m%d")

        self.metadata.metainfo.metd.text = date
        self.metadata.idinfo.citation.citeinfo.pubdate.text = date

    def update_time_period(self, start, end):
        """
        Update time period start and end times

        :param start: DESCRIPTION
        :type start: TYPE
        :param end: DESCRIPTION
        :type end: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.metadata.idinfo.timeperd.timeinfo.rngdates.clear_children()
        xml_utils.XMLNode(
            tag="begdate",
            text=xml_utils.format_date(start),
            parent_node=self.metadata.idinfo.timeperd.timeinfo.rngdates,
            index=0,
        )

        xml_utils.XMLNode(
            tag="begtime",
            text=xml_utils.format_time(start),
            parent_node=self.metadata.idinfo.timeperd.timeinfo.rngdates,
            index=1,
        )

        xml_utils.XMLNode(
            tag="enddate",
            text=xml_utils.format_date(end),
            parent_node=self.metadata.idinfo.timeperd.timeinfo.rngdates,
            index=2,
        )

        xml_utils.XMLNode(
            tag="endtime",
            text=xml_utils.format_time(end),
            parent_node=self.metadata.idinfo.timeperd.timeinfo.rngdates,
            index=3,
        )

    def update_child(self, child_item):
        """
        Update information from a child item
        """
        self.metadata.distinfo.resdesc.text = child_item["link"]["url"]

    def update_shp_attributes(self, df):
        """
        Update the bounds on shapefile attributes from the survey dataframe

        :param df: DESCRIPTION
        :type df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        for attr in self.metadata.eainfo.detailed.attr:
            label = attr.attrlabl.text
            try:
                low = df[label].min()
                high = df[label].max()
                try:
                    attr.attrdomv.rdom.rdommin.text = str(low)
                    attr.attrdomv.rdom.rdommax.text = str(high)
                except AttributeError as error:
                    # print(f"Attribute {label} does not have rdom, {error}")
                    pass

            except KeyError:
                # print(f"could not find {label} in dataframe, skipping")
                pass
            except TypeError as error:
                # print(f"Could convert {label}, {error}")
                pass

    def update_with_station(self, station):
        """
        Update places where {STATION} is set as a place holder in the child
        item.

        title, abstract, files names,

        Parameters
        ----------
        station : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # add station name to title and abstract
        self.metadata.idinfo.citation.citeinfo.title.text = (
            self.metadata.idinfo.citation.citeinfo.title.text.replace(
                "{STATION}", station
            )
        )

        self.metadata.idinfo.descript.abstract.text = (
            self.metadata.idinfo.descript.abstract.text.replace("{STATION}", station)
        )

        # add list of files
        # add list of files
        self.metadata.idinfo.descript.supplinf.text = (
            self.metadata.idinfo.descript.supplinf.text.replace(
                "{STATION_FILES}",
                "\n\t\t\t".join(
                    [
                        f"{station}.edi",
                        f"{station}.png",
                        f"{station}.h5",
                    ]
                ),
            )
        )

        for ii in range(3):
            self.metadata.eainfo.overview[
                ii
            ].eaover.text = self.metadata.eainfo.overview[ii].eaover.text.replace(
                "{STATION}", station
            )
            self.metadata.eainfo.overview[
                ii
            ].eadetcit.text = self.metadata.eainfo.overview[ii].eadetcit.text.replace(
                "{STATION}", station
            )

    def update_metadate(self):
        """
        Update metadate

        Parameters
        ----------
        mdate : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        now = datetime.datetime.utcnow()
        now_str = now.strftime("%Y%m%d")

        self.metadata.metainfo.metd.text = now_str
