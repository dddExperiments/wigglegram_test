/**
 * Match I/O Utilities
 * Supports VisualSFM matches.txt format
 */

import { downloadFile } from './feature-io.js';

/**
 * Serializes matches into VisualSFM format
 * @param {string} name1 Filename of first image
 * @param {string} name2 Filename of second image
 * @param {Array} matches List of [index1, index2] pairs
 * @returns {string}
 */
export function serializeMatchesVisualSFM(name1, name2, matches) {
    let content = `${name1}\n`;
    content += `${name2}\n`;
    content += `${matches.length}\n`;
    for (const pair of matches) {
        content += `${pair[0]} ${pair[1]}\n`;
    }
    content += '\n'; // Blank line separator
    return content;
}

/**
 * Download matches in VisualSFM format
 * @param {string} name1 
 * @param {string} name2 
 * @param {Array} matches 
 * @param {string} filename 
 */
export function downloadMatchesAsVisualSFM(name1, name2, matches, filename = 'matches.txt') {
    const content = serializeMatchesVisualSFM(name1, name2, matches);
    downloadFile(content, filename);
}

/**
 * Serializes a full set of matches for multiple pairs
 * @param {Array} matchResults List of { name1, name2, matches }
 * @returns {string}
 */
export function serializeAllMatches(matchResults) {
    let content = '';
    for (const res of matchResults) {
        content += serializeMatchesVisualSFM(res.name1, res.name2, res.matches);
    }
    return content;
}
