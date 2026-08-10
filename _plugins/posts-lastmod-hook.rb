#!/usr/bin/env ruby
# frozen_string_literal: true

# Batch last-modified timestamps for posts.
#
# Chirpy starter used one `git rev-list` + optional `git log` *per post*
# (O(n) process spawns). With 100+ posts that dominates build time.
#
# This version walks the log once and maps path -> latest author date:
#   O(1) subprocesses, O(C) work over commits that touch _posts/.

module ChirpyLastmod
  module_function

  def cache
    @cache ||= build_cache
  end

  def build_cache
    map = {}
    return map unless File.directory?('.git')

    # %aI = author date, strict ISO-8601; empty line separates commits.
    # --name-only lists paths changed in that commit.
    out = `git log --name-only --pretty=format:%aI -- _posts 2>/dev/null`
    return map if out.nil? || out.empty?

    current_date = nil
    out.each_line do |line|
      line = line.strip
      next if line.empty?

      if line.match?(/\A\d{4}-\d{2}-\d{2}T/)
        current_date = line
        next
      end

      next unless current_date
      next unless line.start_with?('_posts/')
      # First time we see a path is the newest commit (git log is newest-first).
      map[line] ||= current_date
    end

    map
  rescue StandardError
    {}
  end
end

Jekyll::Hooks.register :posts, :post_init do |post|
  # post.path may be absolute; normalize to repo-relative _posts/...
  rel = post.path.to_s.sub(%r{\A.*?(_posts/)}, '\1')
  lastmod = ChirpyLastmod.cache[rel]
  post.data['last_modified_at'] = lastmod if lastmod
end
