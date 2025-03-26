"""
Ontraport API data loader for fetching various data types from Ontraport.
Handles contacts, deals, companies, tasks, and pages with filtering capabilities.
"""

import os
from typing import Dict, List, Optional, Any, Iterator, Union
import requests
from datetime import datetime
import pandas as pd
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from pathlib import Path
import json
from ..types import BasePlugin, DataCategory, DataContainer

# Load environment variables
load_dotenv()
AVAILABLE_ENDPOINTS = ["Contacts", "Deals", "Companies", "Tasks", "Pages"]


class OntraportDownloader(ABC):
    """Base class for Ontraport API interactions."""

    BASE_URL = "https://api.ontraport.com/1"
    DEFAULT_PAGE_SIZE = 50

    def __init__(self, api_key: str, app_id: str, endpoint: Optional[str] = None):
        """
        Initialize the loader with API credentials.

        Args:
            api_key: Ontraport API key
            app_id: Ontraport App ID
            endpoint: Optional endpoint to use (Contacts, Deals, etc.)
        """
        self.headers = {
            "Api-Key": api_key,
            "Api-Appid": app_id,
            "Content-Type": "application/json",
        }
        # if endpoint not in AVAILABLE_ENDPOINTS and endpoint is not None:
        #     raise ValueError(f"Invalid endpoint: {endpoint}")
        self.endpoint = endpoint

    def info(self, search_term: str) -> Dict:
        """
        Get contact information using a search term.

        Args:
            search_term: Term to search for (name, email, etc.)

        Returns:
            Dictionary containing contact information
        """
        if self.endpoint != "Contacts":
            raise ValueError("info method is only available for Contacts endpoint")

        try:
            url = f"{self.BASE_URL}/{self.endpoint}/getInfo"
            response = requests.get(
                url, headers=self.headers, params={"search": search_term}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting contact info: {str(e)}")
            return {}

    def _make_request(
        self, endpoint: str, params: Optional[Dict] = None, paginate: bool = True
    ) -> Dict:
        """
        Make an API request to Ontraport.

        Args:
            endpoint: API endpoint to call
            params: Optional query parameters
            paginate: Whether to include pagination parameters

        Returns:
            API response as dictionary
        """
        url = f"{self.BASE_URL}/{endpoint}"
        if params is None:
            params = {}

        if paginate and "range" not in params:
            params["range"] = f"0,{self.DEFAULT_PAGE_SIZE}"

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {endpoint}: {str(e)}")
            return {"data": []}

    def get_total_records(self, params: Optional[Dict] = None) -> int:
        """
        Get total number of records available for the query using the getInfo endpoint.

        Args:
            params: Optional query parameters to filter the count

        Returns:
            Total number of records
        """
        if not self.endpoint:
            return 0

        try:
            url = f"{self.BASE_URL}/{self.endpoint}/getInfo"
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            if (
                isinstance(data, dict)
                and isinstance(data.get("data"), dict)
                and "count" in data["data"]
            ):
                return int(data["data"]["count"])

            print(f"Warning: Unexpected response format from getInfo endpoint: {data}")
            return self.DEFAULT_PAGE_SIZE

        except requests.exceptions.RequestException as e:
            print(f"Error getting total records for {self.endpoint}: {str(e)}")
            return self.DEFAULT_PAGE_SIZE

    def _fetch_all_pages(self, params: Optional[Dict] = None) -> Iterator[Dict]:
        """
        Fetch all pages of data using pagination.

        Args:
            params: Optional query parameters

        Yields:
            Each page of data as a dictionary
        """
        if params is None:
            params = {}

        total_records = self.get_total_records(params)
        print(f"Total records to fetch: {total_records}")

        start = 0
        while start < total_records:
            page_params = params.copy()
            page_params["start"] = start
            print(f"Fetching records {start} to {start + self.DEFAULT_PAGE_SIZE}")

            response = self._make_request(self.endpoint, page_params, paginate=False)

            if not response or (
                isinstance(response, dict) and not response.get("data")
            ):
                break

            yield response
            start += self.DEFAULT_PAGE_SIZE

    def fetch_data(self, filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Fetch all data from Ontraport API with pagination.

        Args:
            filters: Optional filters to apply to the query

        Returns:
            DataFrame containing all fetched data
        """
        all_data = []
        total_fetched = 0

        for page in self._fetch_all_pages(filters):
            if isinstance(page, dict) and "data" in page:
                data = page["data"]
                if isinstance(data, list):
                    all_data.extend(data)
                    total_fetched += len(data)
                else:
                    all_data.append(data)
                    total_fetched += 1
            elif isinstance(page, list):
                all_data.extend(page)
                total_fetched += len(page)

        df = pd.DataFrame(all_data) if all_data else pd.DataFrame()
        print(f"Total records loaded: {total_fetched}")
        return df


class OntraportLoader(BasePlugin):
    """Plugin for loading Ontraport data from either API (via downloader) or saved files."""

    name = "ontraport_loader"
    supported_extensions = [".json", ".csv"]
    data_category = DataCategory.TABULAR

    def __init__(self):
        """Initialize the loader."""
        self._downloader = None

    def load(self, file_path: Union[Path, str], **kwargs) -> DataContainer:
        """
        Load Ontraport data from a file or API via downloader.

        Args:
            file_path: Path to a saved file or endpoint name for API
            **kwargs: Additional arguments including:
                - downloader: Required OntraportDownloader instance for API operations
                - filters: Optional filters for API requests

        Returns:
            DataContainer with the loaded data and metadata

        Raises:
            ValueError: If trying to load from API without providing a downloader
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # If file_path is an actual file, load from file
        if file_path.exists():
            if file_path.suffix not in self.supported_extensions:
                raise ValueError(f"Unsupported file extension: {file_path.suffix}")

            try:
                if file_path.suffix == ".json":
                    with open(file_path) as f:
                        data = json.load(f)
                    df = pd.DataFrame(data.get("data", []))
                else:  # .csv
                    df = pd.read_csv(file_path)

                metadata = {
                    "category": self.data_category,
                    "source": str(file_path),
                    "rows": len(df),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "loaded_from": "file",
                }
                return DataContainer(data=df, metadata=metadata)

            except Exception as e:
                raise ValueError(f"Error loading file: {e}")

        # For API operations, require a downloader instance
        downloader = kwargs.get("downloader")
        if not downloader:
            raise ValueError("Downloader instance is required for API operations")

        # Set the endpoint and fetch data
        endpoint = file_path.stem
        # if endpoint not in AVAILABLE_ENDPOINTS:
        #     raise ValueError(f"Invalid endpoint: {endpoint}")

        downloader.endpoint = endpoint
        filters = kwargs.get("filters")

        try:
            df = downloader.fetch_data(filters)
            metadata = {
                "category": self.data_category,
                "source": f"Ontraport API - {endpoint}",
                "rows": len(df),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "loaded_from": "api",
                "endpoint": endpoint,
                "filters": filters,
            }
            return DataContainer(data=df, metadata=metadata)

        except Exception as e:
            raise ValueError(f"Error loading from API: {e}")
