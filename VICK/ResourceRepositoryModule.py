import os
import hashlib
import time


class ResourceRepository:
    def __init__(self, directory):
        self.directory = directory
        self.resources = {}
        self._scan_directory()

    def _scan_directory(self):
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            if os.path.isfile(filepath):
                resource_id = self._generate_resource_id(filepath)
                self.resources[resource_id] = {
                    'path': filepath,
                    'metadata': {},
                    'tags': [],
                    'versions': []
                }

    def _generate_resource_id(self, filepath):
        with open(filepath, "rb") as file:
            content = file.read()
        md5_hash = hashlib.md5(content).hexdigest()
        return md5_hash

    def get_resource(self, resource_id):
        return self.resources.get(resource_id, None)

    def _read_file(self, file_path):
        # Read the contents of a file
        with open(file_path, 'r') as file:
            return file.read()

    def _write_file(self, content, file_path):
        # Write content to a file
        with open(file_path, 'w') as file:
            file.write(content)

    def _get_file_extension(self, resource_type):
        # Map resource types to file extensions
        file_extension_map = {
            'text': '.txt',
            'code': '.py',
            'json': '.json'
            # Add more mappings as needed
        }
        if resource_type in file_extension_map:
            return file_extension_map[resource_type]
        else:
            return ''

    def _generate_unique_id(self, file_extension):
        # Generate a unique identifier for the resource
        # This can be based on timestamps, random strings, or any other desired approach
        # Here, we use a simple timestamp-based identifier
        timestamp = str(int(time.time() * 1000))
        return f'resource_{timestamp}{file_extension}'

    def store_resource(self, resource, resource_type, metadata=None, tags=None):
        # Store the given resource in the repository
        file_extension = self._get_file_extension(resource_type)
        resource_id = self._generate_unique_id(file_extension)
        resource_path = os.path.join(self.directory, resource_id)
        self._write_file(resource, resource_path)

        # Update resource metadata
        if metadata is not None:
            self.resources[resource_id]['metadata'] = metadata

        # Update resource tags
        if tags is not None:
            self.resources[resource_id]['tags'] = tags

        # Add to versions history
        self.resources[resource_id]['versions'].append({
            'timestamp': int(time.time()),
            'path': resource_path
        })

        return resource_id

    def get_resource_metadata(self, resource_id):
        resource = self.resources.get(resource_id, None)
        if resource is not None:
            return resource['metadata']
        return None

    def get_resource_tags(self, resource_id):
        resource = self.resources.get(resource_id, None)
        if resource is not None:
            return resource['tags']
        return None

    def search_resources(self, keywords=None, metadata=None, tags=None):
        results = []
        for resource_id, resource in self.resources.items():
            if keywords is not None:
                if self._check_keywords_in_resource(keywords, resource):
                    results.append(resource_id)
            if metadata is not None:
                if self._check_metadata_in_resource(metadata, resource):
                    results.append(resource_id)
            if tags is not None:
                if self._check_tags_in_resource(tags, resource):
                    results.append(resource_id)
        return results

    def _check_keywords_in_resource(self, keywords, resource):
        content = self._read_file(resource['path'])
        for keyword in keywords:
            if keyword.lower() not in content.lower():
                return False
        return True

    def _check_metadata_in_resource(self, metadata, resource):
        for key, value in metadata.items():
            if key in resource['metadata'] and resource['metadata'][key] == value:
                return True
        return False

    def _check_tags_in_resource(self, tags, resource):
        for tag in tags:
            if tag in resource['tags']:
                return True
        return False

    def get_resource_versions(self, resource_id):
        resource = self.resources.get(resource_id, None)
        if resource is not None:
            return resource['versions']
        return None

    def preview_resource(self, resource_id):
        resource = self.resources.get(resource_id, None)
        if resource is not None:
            content = self._read_file(resource['path'])
            return content
        return None

    def delete_resource(self, resource_id):
        resource = self.resources.get(resource_id, None)
        if resource is not None:
            os.remove(resource['path'])
            del self.resources[resource_id]
            return True
        return False

    def get_all_resources(self):
        return list(self.resources.keys())

    def archive_resources(self, max_age_days):
        current_time = int(time.time())
        for resource_id, resource in self.resources.items():
            versions = resource['versions']
            if len(versions) > 0:
                latest_timestamp = versions[-1]['timestamp']
                if (current_time - latest_timestamp) > (max_age_days * 24 * 60 * 60):
                    self.delete_resource(resource_id)
                    continue


# Example usage:

# Create a resource repository
repository = ResourceRepository(directory='/path/to/repository')

# Store a resource
resource = 'Some text content'
resource_type = 'text'
metadata = {'author': 'John Doe', 'created_at': '2023-06-01'}
tags = ['example', 'text']
resource_id = repository.store_resource(resource, resource_type, metadata=metadata, tags=tags)

# Get resource metadata and tags
metadata = repository.get_resource_metadata(resource_id)
tags = repository.get_resource_tags(resource_id)

# Search for resources
keywords = ['text', 'example']
search_results = repository.search_resources(keywords=keywords)

# Get resource versions
versions = repository.get_resource_versions(resource_id)

# Preview a resource
preview_content = repository.preview_resource(resource_id)

# Delete a resource
deleted = repository.delete_resource(resource_id)

# Get all resources in the repository
all_resources = repository.get_all_resources()

# Archive outdated resources (e.g., older than 30 days)
repository.archive_resources(max_age_days=30)
